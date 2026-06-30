//! Trunk output heads: `DistogramModule` and `BFactorModule`.
//!
//! Reference: `boltz-reference/src/boltz/model/modules/trunkv2.py`

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Tensor};

/// Predicted distogram logits from the pair representation `z`.
///
/// Python: `DistogramModule(token_z, num_bins, num_distograms=1)`.
#[derive(Module, Debug)]
pub struct DistogramModule<B: Backend> {
    distogram: Linear<B>,
    #[module(skip)]
    num_distograms: usize,
    #[module(skip)]
    num_bins: usize,
}

impl<B: Backend> DistogramModule<B> {
    pub fn new(
        device: &Device<B>,
        token_z: usize,
        num_bins: usize,
        num_distograms: Option<usize>,
    ) -> Self {
        let num_distograms = num_distograms.unwrap_or(1);
        let distogram = LinearConfig::new(token_z, num_distograms * num_bins).init(device);
        Self {
            distogram,
            num_distograms,
            num_bins,
        }
    }

    /// `z + z^T` → linear → reshape `[B, N, N, num_distograms, num_bins]`.
    pub fn forward(&self, z: Tensor<B, 4>) -> Tensor<B, 5> {
        let z_sym = z.clone() + z.swap_dims(1, 2);
        let out = self.distogram.forward(z_sym);
        let [b, n, _, _] = out.dims();
        out.reshape([b, n, n, self.num_distograms, self.num_bins])
    }

    pub fn num_bins(&self) -> usize {
        self.num_bins
    }

    pub fn num_distograms(&self) -> usize {
        self.num_distograms
    }
}

/// Predicted B-factor histogram from the single representation `s`.
///
/// Python: `BFactorModule(token_s, num_bins)`.
#[derive(Module, Debug)]
pub struct BFactorModule<B: Backend> {
    bfactor: Linear<B>,
    #[module(skip)]
    num_bins: usize,
}

impl<B: Backend> BFactorModule<B> {
    pub fn new(device: &Device<B>, token_s: usize, num_bins: usize) -> Self {
        let bfactor = LinearConfig::new(token_s, num_bins).init(device);
        Self { bfactor, num_bins }
    }

    /// `s → linear → [B, N, num_bins]`.
    pub fn forward(&self, s: Tensor<B, 3>) -> Tensor<B, 3> {
        self.bfactor.forward(s)
    }

    pub fn num_bins(&self) -> usize {
        self.num_bins
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn distogram_forward_shape() {
        let device = Default::default();
        let token_z = 32_usize;
        let num_bins = 64_usize;
        let b = 2_usize;
        let n = 8_usize;
        let m = DistogramModule::<B>::new(&device, token_z, num_bins, None);
        let z = Tensor::<B, 4>::random(
            [b, n, n, token_z],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let out = m.forward(z);
        assert_eq!(out.dims(), [b, n, n, 1, num_bins]);
    }

    #[test]
    fn distogram_multi_forward_shape() {
        let device = Default::default();
        let token_z = 32_usize;
        let num_bins = 64_usize;
        let num_distograms = 3_usize;
        let b = 2_usize;
        let n = 8_usize;
        let m = DistogramModule::<B>::new(&device, token_z, num_bins, Some(num_distograms));
        let z = Tensor::<B, 4>::random(
            [b, n, n, token_z],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let out = m.forward(z);
        assert_eq!(out.dims(), [b, n, n, num_distograms, num_bins]);
    }

    #[test]
    fn bfactor_forward_shape() {
        let device = Default::default();
        let token_s = 64_usize;
        let num_bins = 64_usize;
        let b = 2_usize;
        let n = 8_usize;
        let m = BFactorModule::<B>::new(&device, token_s, num_bins);
        let s = Tensor::<B, 3>::random(
            [b, n, token_s],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let out = m.forward(s);
        assert_eq!(out.dims(), [b, n, num_bins]);
    }
}
