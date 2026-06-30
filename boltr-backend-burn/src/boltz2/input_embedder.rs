//! Boltz `InputEmbedder` — token tail (residue type + MSA profile linears).
//!
//! Full atom stack is Phase 2; this module covers `new_tail_only` / `forward_with_atom_repr`.

use burn::module::Module;
use burn::nn::Linear;
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Tensor};

use crate::burn_compat::linear_no_bias;

/// `len(const.tokens)` in Boltz — must match `boltr_io::boltz_const::NUM_TOKENS`.
pub const BOLTZ_NUM_TOKENS: usize = 33;
pub const BOLTZ_MSA_PROFILE_IN: usize = BOLTZ_NUM_TOKENS + 1;

#[derive(Module, Debug)]
pub struct InputEmbedder<B: Backend> {
    res_type_encoding: Linear<B>,
    msa_profile_encoding: Linear<B>,
    token_s: usize,
}

impl<B: Backend> InputEmbedder<B> {
    pub fn new_tail_only(device: &Device<B>, token_s: usize) -> Self {
        Self {
            res_type_encoding: linear_no_bias(device, BOLTZ_NUM_TOKENS, token_s),
            msa_profile_encoding: linear_no_bias(device, BOLTZ_MSA_PROFILE_IN, token_s),
            token_s,
        }
    }

    pub fn token_s(&self) -> usize {
        self.token_s
    }

    pub fn forward_with_atom_repr(
        &self,
        atom_attn_out: Tensor<B, 3>,
        res_type: Tensor<B, 3>,
        profile: Tensor<B, 3>,
        deletion_mean: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let dm = deletion_mean.unsqueeze_dim::<3>(2);
        let msa_in = Tensor::cat(vec![profile, dm], 2);
        atom_attn_out
            + self.res_type_encoding.forward(res_type)
            + self.msa_profile_encoding.forward(msa_in)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn forward_with_atom_repr_shapes() {
        let device = Default::default();
        let token_s = 384;
        let emb = InputEmbedder::<B>::new_tail_only(&device, token_s);
        let a = Tensor::<B, 3>::random(
            [2, 9, token_s],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let res = Tensor::<B, 3>::random(
            [2, 9, BOLTZ_NUM_TOKENS],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let prof = Tensor::<B, 3>::random(
            [2, 9, BOLTZ_NUM_TOKENS],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let del = Tensor::<B, 2>::random(
            [2, 9],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let s = emb.forward_with_atom_repr(a, res, prof, del);
        assert_eq!(s.dims(), [2, 9, token_s]);
    }
}
