//! `DiffusionConditioning`: pre-computes the atom encoder output, pair biases,
//! and token-transformer bias that the score model consumes at every denoising step.
//!
//! Reference: `boltz-reference/src/boltz/model/modules/diffusion_conditioning.py`

use burn::module::Module;
use burn::nn::{LayerNorm, Linear};
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Tensor};

use crate::burn_compat::{layer_norm_1d, linear_no_bias};

use super::encoders::{AtomEncoder, AtomEncoderBatchFeats, AtomEncoderFlags, PairwiseConditioning};

/// Pre-computed conditioning tensors consumed by the score model at each step.
pub struct DiffusionConditioningOutput<B: Backend> {
    /// Atom query features `[B, M, atom_s]`.
    pub q: Tensor<B, 3>,
    /// Atom conditioning features `[B, M, atom_s]`.
    pub c: Tensor<B, 3>,
    /// Windowed pair features `[B, K, W, H, atom_z]`.
    pub p: Tensor<B, 5>,
    /// Indexing matrix for windowed key construction.
    pub indexing_matrix: Tensor<B, 2>,
    /// Atom encoder bias (all depths concatenated) `[B, K, W, H, total_enc_heads]`.
    pub atom_enc_bias: Tensor<B, 5>,
    /// Atom decoder bias (all depths concatenated) `[B, K, W, H, total_dec_heads]`.
    pub atom_dec_bias: Tensor<B, 5>,
    /// Token transformer bias (all depths concatenated) `[B, N, N, total_trans_heads]`.
    pub token_trans_bias: Tensor<B, 4>,
}

#[derive(Module, Debug)]
pub struct LayerNormLinear<B: Backend> {
    norm: LayerNorm<B>,
    linear: Linear<B>,
}

impl<B: Backend> LayerNormLinear<B> {
    fn new(device: &Device<B>, in_dim: usize, out_dim: usize) -> Self {
        Self {
            norm: layer_norm_1d(device, in_dim),
            linear: linear_no_bias(device, in_dim, out_dim),
        }
    }

    fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        self.linear.forward(self.norm.forward(x))
    }

    fn forward_z(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.linear.forward(self.norm.forward(x))
    }
}

/// `DiffusionConditioning` module — runs once per forward / sample call (not per diffusion step).
#[derive(Module, Debug)]
pub struct DiffusionConditioning<B: Backend> {
    pairwise_conditioner: PairwiseConditioning<B>,
    atom_encoder: AtomEncoder<B>,
    atom_enc_proj_z: Vec<LayerNormLinear<B>>,
    atom_dec_proj_z: Vec<LayerNormLinear<B>>,
    token_trans_proj_z: Vec<LayerNormLinear<B>>,
}

impl<B: Backend> DiffusionConditioning<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device<B>,
        token_s: usize,
        token_z: usize,
        atom_s: usize,
        atom_z: usize,
        atoms_per_window_queries: usize,
        atoms_per_window_keys: usize,
        atom_encoder_depth: usize,
        atom_encoder_heads: usize,
        token_transformer_depth: usize,
        token_transformer_heads: usize,
        atom_decoder_depth: usize,
        atom_decoder_heads: usize,
        atom_feature_dim: usize,
        conditioning_transition_layers: usize,
        atom_flags: AtomEncoderFlags,
    ) -> Self {
        let pairwise_conditioner = PairwiseConditioning::new(
            device,
            token_z,
            token_z,
            conditioning_transition_layers,
            2,
        );

        let atom_encoder = AtomEncoder::new(
            device,
            atom_s,
            atom_z,
            token_s,
            token_z,
            atoms_per_window_queries,
            atoms_per_window_keys,
            atom_feature_dim,
            true,
            atom_flags,
        );

        let atom_enc_proj_z = (0..atom_encoder_depth)
            .map(|_| LayerNormLinear::new(device, atom_z, atom_encoder_heads))
            .collect();
        let atom_dec_proj_z = (0..atom_decoder_depth)
            .map(|_| LayerNormLinear::new(device, atom_z, atom_decoder_heads))
            .collect();
        let token_trans_proj_z = (0..token_transformer_depth)
            .map(|_| LayerNormLinear::new(device, token_z, token_transformer_heads))
            .collect();

        Self {
            pairwise_conditioner,
            atom_encoder,
            atom_enc_proj_z,
            atom_dec_proj_z,
            token_trans_proj_z,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        s_trunk: Tensor<B, 3>,
        z_trunk: Tensor<B, 4>,
        relative_position_encoding: Tensor<B, 4>,
        ref_pos: Tensor<B, 3>,
        ref_charge: Tensor<B, 2>,
        ref_element: Tensor<B, 3>,
        atom_pad_mask: Tensor<B, 2>,
        ref_space_uid: Tensor<B, 2, burn::tensor::Int>,
        atom_to_token: Tensor<B, 3>,
        batch: Option<&AtomEncoderBatchFeats<'_, B>>,
    ) -> DiffusionConditioningOutput<B> {
        let z = self
            .pairwise_conditioner
            .forward(z_trunk, relative_position_encoding);

        let (q, c, p, indexing_matrix) = self.atom_encoder.forward(
            ref_pos,
            ref_charge,
            ref_element,
            atom_pad_mask,
            ref_space_uid,
            atom_to_token,
            Some(s_trunk),
            Some(z.clone()),
            batch,
        );

        let atom_enc_bias = Tensor::cat(
            self.atom_enc_proj_z
                .iter()
                .map(|proj| proj.forward(p.clone()))
                .collect::<Vec<_>>(),
            4,
        );

        let atom_dec_bias = Tensor::cat(
            self.atom_dec_proj_z
                .iter()
                .map(|proj| proj.forward(p.clone()))
                .collect::<Vec<_>>(),
            4,
        );

        let token_trans_bias = Tensor::cat(
            self.token_trans_proj_z
                .iter()
                .map(|proj| proj.forward_z(z.clone()))
                .collect::<Vec<_>>(),
            3,
        );

        DiffusionConditioningOutput {
            q,
            c,
            p,
            indexing_matrix,
            atom_enc_bias,
            atom_dec_bias,
            token_trans_bias,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Int;

    type B = NdArray;

    #[test]
    fn diffusion_conditioning_output_shapes() {
        let device = Default::default();

        let token_s = 32_usize;
        let token_z = 16_usize;
        let atom_s = 16_usize;
        let atom_z = 8_usize;
        let w = 4_usize;
        let h = 8_usize;
        let enc_depth = 2_usize;
        let enc_heads = 2_usize;
        let dec_depth = 2_usize;
        let dec_heads = 2_usize;
        let trans_depth = 2_usize;
        let trans_heads = 4_usize;
        let atom_feat_dim = 3 + 1 + 4;

        let atom_flags = AtomEncoderFlags {
            num_elements: 4,
            use_no_atom_char: true,
            use_atom_backbone_feat: false,
            use_residue_feats_atoms: false,
            backbone_feat_dim: 17,
            num_tokens: 33,
        };
        assert_eq!(atom_flags.expected_atom_feature_dim(), atom_feat_dim);
        let dc = DiffusionConditioning::<B>::new(
            &device,
            token_s,
            token_z,
            atom_s,
            atom_z,
            w,
            h,
            enc_depth,
            enc_heads,
            trans_depth,
            trans_heads,
            dec_depth,
            dec_heads,
            atom_feat_dim,
            2,
            atom_flags,
        );

        let b = 1_usize;
        let n_tokens = 4_usize;
        let n_atoms = 8_usize;

        let s_trunk = Tensor::<B, 3>::random(
            [b, n_tokens, token_s],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let z_trunk = Tensor::<B, 4>::random(
            [b, n_tokens, n_tokens, token_z],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let rel_pos = Tensor::<B, 4>::random(
            [b, n_tokens, n_tokens, token_z],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let ref_pos = Tensor::<B, 3>::random(
            [b, n_atoms, 3],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let ref_charge = Tensor::<B, 2>::random(
            [b, n_atoms],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let ref_element = Tensor::<B, 3>::random(
            [b, n_atoms, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let atom_pad_mask = Tensor::<B, 2>::ones([b, n_atoms], &device);
        let ref_space_uid = Tensor::<B, 2, Int>::zeros([b, n_atoms], &device);
        let atom_to_token = Tensor::<B, 3>::zeros([b, n_atoms, n_tokens], &device);

        let out = dc.forward(
            s_trunk,
            z_trunk,
            rel_pos,
            ref_pos,
            ref_charge,
            ref_element,
            atom_pad_mask,
            ref_space_uid,
            atom_to_token,
            None,
        );

        assert_eq!(out.q.dims()[0], b);
        assert_eq!(out.q.dims()[1], n_atoms);
        assert_eq!(out.q.dims()[2], atom_s);

        assert_eq!(out.c.dims()[0], b);
        assert_eq!(out.c.dims()[1], n_atoms);

        let k = n_atoms / w;
        assert_eq!(
            out.atom_enc_bias.dims(),
            [b, k, w, h, enc_depth * enc_heads]
        );
        assert_eq!(
            out.atom_dec_bias.dims(),
            [b, k, w, h, dec_depth * dec_heads]
        );
        assert_eq!(
            out.token_trans_bias.dims(),
            [b, n_tokens, n_tokens, trans_depth * trans_heads]
        );
    }
}
