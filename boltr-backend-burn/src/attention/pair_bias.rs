//! Attention pair bias layer (Boltz2 / attentionv2.py).

use burn::module::Module;
use burn::nn::{LayerNorm, Linear, LinearConfig};
use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Tensor};

use crate::burn_compat::{layer_norm_1d, linear_no_bias};
use crate::tensor_ops::{einsum_bihd_bjhd_bhij, einsum_bhij_bjhd_bihd, mask_2d_from_1d};

#[derive(Module, Debug)]
pub struct AttentionPairBiasV2<B: Backend> {
    c_s: usize,
    num_heads: usize,
    head_dim: usize,
    inf: f64,
    proj_q: Linear<B>,
    proj_k: Linear<B>,
    proj_v: Linear<B>,
    proj_g: Linear<B>,
    proj_o: Linear<B>,
    proj_z: Option<(LayerNorm<B>, Linear<B>)>,
    compute_pair_bias: bool,
}

impl<B: Backend> AttentionPairBiasV2<B> {
    pub fn new(
        device: &Device<B>,
        c_s: usize,
        c_z: Option<usize>,
        num_heads: Option<usize>,
        inf: Option<f64>,
    ) -> Self {
        let num_heads = num_heads.unwrap_or(16);
        assert_eq!(c_s % num_heads, 0, "c_s must divide num_heads");
        let head_dim = c_s / num_heads;
        let compute_pair_bias = c_z.is_some();
        let proj_z = c_z.map(|cz| {
            (
                layer_norm_1d(device, cz),
                linear_no_bias(device, cz, num_heads),
            )
        });
        Self {
            c_s,
            num_heads,
            head_dim,
            inf: inf.unwrap_or(1e6),
            proj_q: LinearConfig::new(c_s, c_s).init(device),
            proj_k: LinearConfig::new(c_s, c_s)
                .with_bias(false)
                .init(device),
            proj_v: LinearConfig::new(c_s, c_s)
                .with_bias(false)
                .init(device),
            proj_g: linear_no_bias(device, c_s, c_s),
            proj_o: LinearConfig::new(c_s, c_s)
                .with_bias(false)
                .init(device),
            proj_z,
            compute_pair_bias,
        }
    }

    pub fn forward(
        &self,
        s: Tensor<B, 3>,
        z: Tensor<B, 4>,
        mask: Tensor<B, 3>,
        k_in: Tensor<B, 3>,
        multiplicity: Option<i64>,
    ) -> Tensor<B, 3> {
        self.forward_with_mask_rank(s, z, mask, k_in, multiplicity)
    }

    pub fn forward_key_mask(
        &self,
        s: Tensor<B, 3>,
        z: Tensor<B, 4>,
        mask: Tensor<B, 2>,
        k_in: Tensor<B, 3>,
        multiplicity: Option<i64>,
    ) -> Tensor<B, 3> {
        let [_b, q_len, _] = s.dims();
        let k_len = k_in.dims()[1];
        let mask_expanded = if q_len != k_len {
            mask.unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(1)
        } else {
            mask_2d_from_1d(mask).unsqueeze_dim::<4>(1)
        };
        self.forward_impl(s, z, mask_expanded, k_in, multiplicity)
    }

    fn forward_with_mask_rank(
        &self,
        s: Tensor<B, 3>,
        z: Tensor<B, 4>,
        mask: Tensor<B, 3>,
        k_in: Tensor<B, 3>,
        multiplicity: Option<i64>,
    ) -> Tensor<B, 3> {
        let mask_expanded = mask.unsqueeze_dim::<4>(1);
        self.forward_impl(s, z, mask_expanded, k_in, multiplicity)
    }

    fn forward_impl(
        &self,
        s: Tensor<B, 3>,
        z: Tensor<B, 4>,
        mask_expanded: Tensor<B, 4>,
        k_in: Tensor<B, 3>,
        multiplicity: Option<i64>,
    ) -> Tensor<B, 3> {
        let multiplicity = multiplicity.unwrap_or(1).max(1) as usize;
        let [b, q_len, _] = s.dims();
        let k_len = k_in.dims()[1];

        let q = self
            .proj_q
            .forward(s.clone())
            .reshape([b, q_len, self.num_heads, self.head_dim]);
        let k = self
            .proj_k
            .forward(k_in.clone())
            .reshape([b, k_len, self.num_heads, self.head_dim]);
        let v = self
            .proj_v
            .forward(k_in.clone())
            .reshape([b, k_len, self.num_heads, self.head_dim]);

        let mut bias: Tensor<B, 4> = if self.compute_pair_bias {
            let (norm, linear) = self.proj_z.as_ref().unwrap();
            let z_proj = norm.forward(z);
            let z_linear = linear.forward(z_proj);
            z_linear.swap_dims(1, 3).swap_dims(2, 3)
        } else if z.dims()[3] == 1 {
            z.squeeze_dim::<3>(3)
                .unsqueeze_dim::<4>(1)
                .repeat(&[1, self.num_heads, 1, 1])
        } else {
            z.swap_dims(1, 3).swap_dims(2, 3)
        };

        if multiplicity > 1 {
            bias = bias.repeat(&[multiplicity, 1, 1, 1]);
        }

        let g = sigmoid(self.proj_g.forward(s.clone()));
        let scale = (self.head_dim as f64).sqrt();
        let mut attn = einsum_bihd_bjhd_bhij(q, k) / scale;
        attn = attn + bias;

        let inv = Tensor::<B, 4>::ones(mask_expanded.dims(), &s.device()) - mask_expanded.clone();
        attn = attn + inv * (-self.inf);

        attn = softmax(attn, 3);
        let o = einsum_bhij_bjhd_bihd(attn, v);
        let o = o.reshape([b, q_len, self.c_s]);
        self.proj_o.forward(o * g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn attention_pair_bias_shape() {
        let device = Default::default();
        let layer = AttentionPairBiasV2::<B>::new(&device, 64, Some(32), Some(4), None);
        let s = Tensor::<B, 3>::random(
            [2, 10, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let z = Tensor::<B, 4>::random(
            [2, 10, 10, 32],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let mask = Tensor::<B, 3>::ones([2, 10, 10], &device);
        let out = layer.forward(s.clone(), z, mask, s, None);
        assert_eq!(out.dims(), [2, 10, 64]);
    }

    #[test]
    fn attention_pair_bias_no_bias_shape() {
        let device = Default::default();
        let layer = AttentionPairBiasV2::<B>::new(&device, 64, None, Some(4), None);
        let s = Tensor::<B, 3>::random(
            [2, 10, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let z = Tensor::<B, 4>::zeros([2, 10, 10, 1], &device);
        let mask = Tensor::<B, 3>::ones([2, 10, 10], &device);
        let out = layer.forward(s.clone(), z, mask, s, None);
        assert_eq!(out.dims(), [2, 10, 64]);
    }
}
