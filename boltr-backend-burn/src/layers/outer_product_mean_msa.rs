//! Outer product mean used inside Boltz2 `MSALayer`.

use burn::module::Module;
use burn::nn::{LayerNorm, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Tensor};

use crate::burn_compat::layer_norm_1d;
use crate::tensor_ops::einsum_bsic_bsjd_bijcd;

#[derive(Module, Debug)]
pub struct OuterProductMeanMsa<B: Backend> {
    c_hidden: usize,
    norm: LayerNorm<B>,
    proj_a: Linear<B>,
    proj_b: Linear<B>,
    proj_o: Linear<B>,
}

impl<B: Backend> OuterProductMeanMsa<B> {
    pub fn new(device: &Device<B>, c_in: usize, c_hidden: usize, c_out: usize) -> Self {
        Self {
            c_hidden,
            norm: layer_norm_1d(device, c_in),
            proj_a: LinearConfig::new(c_in, c_hidden)
                .with_bias(false)
                .init(device),
            proj_b: LinearConfig::new(c_in, c_hidden)
                .with_bias(false)
                .init(device),
            proj_o: LinearConfig::new(c_hidden * c_hidden, c_out).init(device),
        }
    }

    pub fn forward(&self, m: Tensor<B, 4>, msa_mask: Tensor<B, 3>) -> Tensor<B, 4> {
        let mask_4 = msa_mask.unsqueeze_dim::<4>(3);
        let m = self.norm.forward(m);
        let a = self.proj_a.forward(m.clone()) * mask_4.clone();
        let b = self.proj_b.forward(m) * mask_4.clone();

        let pair_mask = mask_4.clone().unsqueeze_dim::<5>(2) * mask_4.unsqueeze_dim::<5>(1);

        let z5 = einsum_bsic_bsjd_bijcd(a, b);
        let [batch, i, j, _, _] = z5.dims();
        let z = z5.reshape([batch, i, j, self.c_hidden * self.c_hidden]);
        let num_mask = pair_mask
            .sum_dim(1)
            .sum_dim(2)
            .squeeze_dims::<3>(&[1, 2])
            .unsqueeze_dim::<4>(2)
            .clamp_min(1.0);
        self.proj_o.forward(z / num_mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn outer_product_mean_msa_shape() {
        let device = Default::default();
        let layer = OuterProductMeanMsa::<B>::new(&device, 16, 32, 24);
        let m = Tensor::<B, 4>::random(
            [1, 4, 6, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let mask = Tensor::<B, 3>::ones([1, 4, 6], &device);
        let out = layer.forward(m, mask);
        assert_eq!(out.dims(), [1, 6, 6, 24]);
    }
}
