//! Triangular attention (starting / ending node).

use burn::module::Module;
use burn::nn::{LayerNorm, Linear, LinearConfig};
use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Tensor};

use crate::burn_compat::layer_norm_1d;

#[derive(Module, Debug)]
struct TriangleMha<B: Backend> {
    linear_q: Linear<B>,
    linear_k: Linear<B>,
    linear_v: Linear<B>,
    linear_o: Linear<B>,
    linear_g: Linear<B>,
}

#[derive(Module, Debug)]
pub struct TriangleAttention<B: Backend> {
    layer_norm: LayerNorm<B>,
    linear: Linear<B>,
    mha: TriangleMha<B>,
    c_hidden: usize,
    no_heads: usize,
    starting: bool,
    inf: f64,
}

impl<B: Backend> TriangleAttention<B> {
    pub fn new(
        device: &Device<B>,
        c_in: usize,
        c_hidden: usize,
        no_heads: usize,
        starting: bool,
        inf: f64,
    ) -> Self {
        let mha_dim = c_hidden * no_heads;
        Self {
            layer_norm: layer_norm_1d(device, c_in),
            linear: LinearConfig::new(c_in, no_heads)
                .with_bias(false)
                .init(device),
            mha: TriangleMha {
                linear_q: LinearConfig::new(c_in, mha_dim)
                    .with_bias(false)
                    .init(device),
                linear_k: LinearConfig::new(c_in, mha_dim)
                    .with_bias(false)
                    .init(device),
                linear_v: LinearConfig::new(c_in, mha_dim)
                    .with_bias(false)
                    .init(device),
                linear_o: LinearConfig::new(mha_dim, c_in)
                    .with_bias(false)
                    .init(device),
                linear_g: LinearConfig::new(c_in, mha_dim)
                    .with_bias(false)
                    .init(device),
            },
            c_hidden,
            no_heads,
            starting,
            inf,
        }
    }

    pub fn new_ending_node(
        device: &Device<B>,
        c_in: usize,
        c_hidden: usize,
        no_heads: usize,
        inf: f64,
    ) -> Self {
        Self::new(device, c_in, c_hidden, no_heads, false, inf)
    }

    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        mask: Option<Tensor<B, 3>>,
        _chunk_size: Option<i64>,
        _use_kernels: bool,
    ) -> Tensor<B, 4> {
        let device = x.device();
        let mut x = x;
        let [b, i, j, _] = x.dims();
        let mut mask_t =
            mask.unwrap_or_else(|| Tensor::<B, 3>::ones([b, i, j], &device));

        if !self.starting {
            x = x.swap_dims(1, 2);
            mask_t = mask_t.swap_dims(1, 2);
        }

        x = self.layer_norm.forward(x);
        // [B, I, J] -> [B, I, 1, 1, J] (Python: mask[..., :, None, None, :])
        let mask_expanded = mask_t
            .unsqueeze_dim::<4>(2)
            .unsqueeze_dim::<5>(2);
        let mask_bias = mask_expanded.clone() * self.inf - self.inf;

        // [B, I, J, H] -> [B, H, I, J] -> [B, 1, H, I, J]
        let triangle_bias = self.linear.forward(x.clone());
        let triangle_bias = triangle_bias
            .swap_dims(2, 3)
            .swap_dims(1, 2)
            .unsqueeze_dim::<5>(1);

        let output = self.mha_with_bias(x, mask_bias, triangle_bias);
        if self.starting {
            output
        } else {
            output.swap_dims(1, 2)
        }
    }

    fn mha_with_bias(
        &self,
        x: Tensor<B, 4>,
        mask_bias: Tensor<B, 5>,
        triangle_bias: Tensor<B, 5>,
    ) -> Tensor<B, 4> {
        let [b, i, j, _c_in] = x.dims();
        let h = self.no_heads;
        let d = self.c_hidden;
        let scale = (d as f64).sqrt();

        let q = self.mha.linear_q.forward(x.clone())
            .reshape([b, i, j, h, d])
            .swap_dims(2, 3)
            / scale;
        let k = self
            .mha
            .linear_k
            .forward(x.clone())
            .reshape([b, i, j, h, d])
            .swap_dims(2, 3);
        let v = self
            .mha
            .linear_v
            .forward(x.clone())
            .reshape([b, i, j, h, d])
            .swap_dims(2, 3);

        let mut a = q.matmul(k.clone().swap_dims(3, 4));
        a = a + mask_bias;
        a = a + triangle_bias;
        a = softmax(a, 4);

        let mut o = a.matmul(v);
        o = o.swap_dims(2, 3);
        let g = sigmoid(self.mha.linear_g.forward(x)).reshape([b, i, j, h, d]);
        o = o * g;
        let o = o.reshape([b, i, j, h * d]);
        self.mha.linear_o.forward(o)
    }
}

pub type TriangleAttentionStartingNode<B> = TriangleAttention<B>;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn triangle_attention_shapes() {
        let device = Default::default();
        let c_in = 128;
        let layer = TriangleAttention::<B>::new(&device, c_in, 32, 4, true, 1e9);
        let x = Tensor::<B, 4>::random(
            [2, 10, 10, c_in],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let mask = Tensor::<B, 3>::ones([2, 10, 10], &device);
        let out = layer.forward(x, Some(mask), None, false);
        assert_eq!(out.dims(), [2, 10, 10, c_in]);
    }
}
