//! Transition layer (two-layer MLP with SwiGLU)
//!
//! Reference: boltz-reference/src/boltz/model/layers/transition.py

use burn::module::Module;
use burn::nn::{LayerNorm, Linear, LinearConfig};
use burn::tensor::activation::silu;
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Tensor};

use crate::burn_compat::layer_norm_1d;

/// Transition block: LayerNorm → SiLU(fc1) * fc2 → fc3.
#[derive(Module, Debug)]
pub struct Transition<B: Backend> {
    norm: LayerNorm<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
}

impl<B: Backend> Transition<B> {
    pub fn new(
        device: &Device<B>,
        dim: usize,
        hidden: Option<usize>,
        out_dim: Option<usize>,
    ) -> Self {
        let hidden = hidden.unwrap_or(dim * 4);
        let out_dim = out_dim.unwrap_or(dim);
        Self {
            norm: layer_norm_1d(device, dim),
            fc1: LinearConfig::new(dim, hidden)
                .with_bias(false)
                .init(device),
            fc2: LinearConfig::new(dim, hidden)
                .with_bias(false)
                .init(device),
            fc3: LinearConfig::new(hidden, out_dim)
                .with_bias(false)
                .init(device),
        }
    }

    pub fn forward<const D: usize>(
        &self,
        x: Tensor<B, D>,
        _chunk_size: Option<i64>,
    ) -> Tensor<B, D> {
        let x_normed = self.norm.forward(x);
        let silu_out = silu(self.fc1.forward(x_normed.clone()));
        let gated = silu_out * self.fc2.forward(x_normed);
        self.fc3.forward(gated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn transition_forward_shape() {
        let device = Default::default();
        let dim = 128;
        let layer = Transition::<B>::new(&device, dim, Some(512), None);
        let x = Tensor::<B, 3>::random(
            [2, 10, dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let out = layer.forward(x, None);
        assert_eq!(out.dims(), [2, 10, dim]);
    }
}
