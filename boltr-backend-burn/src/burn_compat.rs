//! Helpers aligned with PyTorch / `boltr-backend-tch` conventions on Burn.

use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Tensor};

pub fn layer_norm_1d<B: Backend>(device: &Device<B>, dim: usize) -> LayerNorm<B> {
    LayerNormConfig::new(dim)
        .with_epsilon((dim as f64) * 1e-5)
        .init(device)
}

pub fn linear_no_bias<B: Backend>(
    device: &Device<B>,
    in_features: usize,
    out_features: usize,
) -> Linear<B> {
    LinearConfig::new(in_features, out_features)
        .with_bias(false)
        .init(device)
}

pub fn layer_norm_no_affine<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    dim: usize,
) -> Tensor<B, D> {
    let eps = (dim as f64) * 1e-5;
    let mean = x.clone().mean_dim(D - 1);
    let var = x.clone().var(D - 1);
    (x - mean) / (var + eps).sqrt()
}

#[derive(Module, Debug)]
pub struct LayerNormWeightOnly<B: Backend> {
    weight: Tensor<B, 1>,
    dim: usize,
}

impl<B: Backend> LayerNormWeightOnly<B> {
    pub fn new(device: &Device<B>, dim: usize) -> Self {
        Self {
            weight: Tensor::ones([dim], device),
            dim,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let eps = (self.dim as f64) * 1e-5;
        let mean = x.clone().mean_dim(2);
        let var = x.clone().var(2);
        let normed = (x - mean) / (var + eps).sqrt();
        let w = self.weight.clone().unsqueeze_dims(&[0, 1]);
        normed * w
    }
}
