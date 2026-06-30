//! Triangular multiplication layers (outgoing / incoming).

use burn::module::Module;
use burn::nn::{LayerNorm, Linear, LinearConfig};
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Tensor};

use crate::burn_compat::layer_norm_1d;
use crate::tensor_ops::{chunk2, einsum_bikd_bjkd_bijd, einsum_bkid_bkjd_bijd};

#[derive(Module, Debug)]
pub struct TriangleMultiplicationOutgoing<B: Backend> {
    norm_in: LayerNorm<B>,
    p_in: Linear<B>,
    g_in: Linear<B>,
    norm_out: LayerNorm<B>,
    p_out: Linear<B>,
    g_out: Linear<B>,
}

impl<B: Backend> TriangleMultiplicationOutgoing<B> {
    pub fn new(device: &Device<B>, dim: usize) -> Self {
        Self {
            norm_in: layer_norm_1d(device, dim),
            p_in: LinearConfig::new(dim, 2 * dim)
                .with_bias(false)
                .init(device),
            g_in: LinearConfig::new(dim, 2 * dim)
                .with_bias(false)
                .init(device),
            norm_out: layer_norm_1d(device, dim),
            p_out: LinearConfig::new(dim, dim)
                .with_bias(false)
                .init(device),
            g_out: LinearConfig::new(dim, dim)
                .with_bias(false)
                .init(device),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        mask: Tensor<B, 3>,
        _use_kernels: bool,
    ) -> Tensor<B, 4> {
        let x_normed = self.norm_in.forward(x);
        let x_in = x_normed.clone();
        let x = self.p_in.forward(x_normed.clone()) * sigmoid(self.g_in.forward(x_normed));
        let mask = mask.unsqueeze_dim::<4>(3);
        let x = x * mask;
        let (a, b) = chunk2(x, 3);
        let x_tri = einsum_bikd_bjkd_bijd(a, b);
        let x_normed_out = self.norm_out.forward(x_tri);
        self.p_out.forward(x_normed_out) * sigmoid(self.g_out.forward(x_in))
    }
}

#[derive(Module, Debug)]
pub struct TriangleMultiplicationIncoming<B: Backend> {
    norm_in: LayerNorm<B>,
    p_in: Linear<B>,
    g_in: Linear<B>,
    norm_out: LayerNorm<B>,
    p_out: Linear<B>,
    g_out: Linear<B>,
}

impl<B: Backend> TriangleMultiplicationIncoming<B> {
    pub fn new(device: &Device<B>, dim: usize) -> Self {
        Self {
            norm_in: layer_norm_1d(device, dim),
            p_in: LinearConfig::new(dim, 2 * dim)
                .with_bias(false)
                .init(device),
            g_in: LinearConfig::new(dim, 2 * dim)
                .with_bias(false)
                .init(device),
            norm_out: layer_norm_1d(device, dim),
            p_out: LinearConfig::new(dim, dim)
                .with_bias(false)
                .init(device),
            g_out: LinearConfig::new(dim, dim)
                .with_bias(false)
                .init(device),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        mask: Tensor<B, 3>,
        _use_kernels: bool,
    ) -> Tensor<B, 4> {
        let x_normed = self.norm_in.forward(x);
        let x_in = x_normed.clone();
        let x = self.p_in.forward(x_normed.clone()) * sigmoid(self.g_in.forward(x_normed));
        let mask = mask.unsqueeze_dim::<4>(3);
        let x = x * mask;
        let (a, b) = chunk2(x, 3);
        let x_tri = einsum_bkid_bkjd_bijd(a, b);
        let x_normed_out = self.norm_out.forward(x_tri);
        self.p_out.forward(x_normed_out) * sigmoid(self.g_out.forward(x_in))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn triangular_outgoing_shape() {
        let device = Default::default();
        let dim = 128;
        let layer = TriangleMultiplicationOutgoing::<B>::new(&device, dim);
        let x = Tensor::<B, 4>::random(
            [2, 10, 10, dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let mask = Tensor::<B, 3>::ones([2, 10, 10], &device);
        let out = layer.forward(x, mask, false);
        assert_eq!(out.dims(), [2, 10, 10, dim]);
    }

    #[test]
    fn triangular_incoming_shape() {
        let device = Default::default();
        let dim = 128;
        let layer = TriangleMultiplicationIncoming::<B>::new(&device, dim);
        let x = Tensor::<B, 4>::random(
            [2, 10, 10, dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let mask = Tensor::<B, 3>::ones([2, 10, 10], &device);
        let out = layer.forward(x, mask, false);
        assert_eq!(out.dims(), [2, 10, 10, dim]);
    }
}
