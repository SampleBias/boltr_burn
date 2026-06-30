//! Pair-weighted averaging (`PairWeightedAveraging` in Boltz).

use burn::module::Module;
use burn::nn::{LayerNorm, Linear};
use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Tensor};

use crate::burn_compat::{layer_norm_1d, linear_no_bias};
use crate::tensor_ops::einsum_bhij_bhsjd_bhsid;

#[derive(Module, Debug)]
pub struct PairWeightedAveraging<B: Backend> {
    c_m: usize,
    c_h: usize,
    num_heads: usize,
    inf: f64,
    norm_m: LayerNorm<B>,
    norm_z: LayerNorm<B>,
    proj_m: Linear<B>,
    proj_g: Linear<B>,
    proj_z: Linear<B>,
    proj_o: Linear<B>,
}

impl<B: Backend> PairWeightedAveraging<B> {
    pub fn new(
        device: &Device<B>,
        c_m: usize,
        c_z: usize,
        c_h: usize,
        num_heads: usize,
        inf: Option<f64>,
    ) -> Self {
        Self {
            c_m,
            c_h,
            num_heads,
            inf: inf.unwrap_or(1e6),
            norm_m: layer_norm_1d(device, c_m),
            norm_z: layer_norm_1d(device, c_z),
            proj_m: linear_no_bias(device, c_m, c_h * num_heads),
            proj_g: linear_no_bias(device, c_m, c_h * num_heads),
            proj_z: linear_no_bias(device, c_z, num_heads),
            proj_o: linear_no_bias(device, c_h * num_heads, c_m),
        }
    }

    pub fn forward(
        &self,
        m: Tensor<B, 4>,
        z: Tensor<B, 4>,
        pair_mask: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        let [b, s, n, _] = m.dims();
        let m = self.norm_m.forward(m);
        let z = self.norm_z.forward(z);

        let v = self
            .proj_m
            .forward(m.clone())
            .reshape([b, s, n, self.num_heads, self.c_h])
            .swap_dims(1, 3)
            .swap_dims(2, 3);

        let mut bias = self.proj_z.forward(z);
        bias = bias.swap_dims(1, 3).swap_dims(2, 3);
        let mask_exp = pair_mask.unsqueeze_dim::<4>(1);
        let inv_mask = Tensor::<B, 4>::ones(mask_exp.dims(), &m.device()) - mask_exp.clone();
        bias = bias + inv_mask * (-self.inf);
        let w = softmax(bias, 3);

        let g = sigmoid(self.proj_g.forward(m));

        let o = einsum_bhij_bhsjd_bhsid(w, v);
        let o = o.swap_dims(1, 2).swap_dims(2, 3).reshape([b, s, n, self.num_heads * self.c_h]);
        self.proj_o.forward(g * o)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn pair_weighted_averaging_shape() {
        let device = Default::default();
        let layer = PairWeightedAveraging::<B>::new(&device, 64, 32, 32, 8, None);
        let m = Tensor::<B, 4>::random(
            [1, 4, 6, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let z = Tensor::<B, 4>::random(
            [1, 6, 6, 32],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let mask = Tensor::<B, 3>::ones([1, 6, 6], &device);
        let out = layer.forward(m, z, mask);
        assert_eq!(out.dims(), [1, 4, 6, 64]);
    }
}
