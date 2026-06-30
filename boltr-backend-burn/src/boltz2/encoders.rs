//! Encoder modules for diffusion conditioning and the score model.
//!
//! Reference: `boltz-reference/src/boltz/model/modules/encodersv2.py`

use std::f64::consts::PI;

use burn::module::Module;
use burn::nn::{LayerNorm, Linear, LinearConfig};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Int, Tensor};

use crate::burn_compat::{layer_norm_1d, linear_no_bias};
use crate::layers::Transition;
use crate::tensor_ops::{einsum_bijd_bwki_bwlj_bwkld, one_hot_2d, repeat_interleave_dim0};

use super::atom_window_keys::{get_indexing_matrix, single_to_keys};
use super::transformers::AtomTransformer;

// ---------------------------------------------------------------------------
// FourierEmbedding  (Algorithm 22)
// ---------------------------------------------------------------------------

/// Frozen random Fourier feature embedding for noise level σ.
#[derive(Module, Debug)]
pub struct FourierEmbedding<B: Backend> {
    proj: Linear<B>,
}

impl<B: Backend> FourierEmbedding<B> {
    pub fn new(device: &Device<B>, dim: usize) -> Self {
        let proj = LinearConfig::new(1, dim).init(device);
        Self { proj }
    }

    /// `times: [B]` → `[B, dim]`.
    pub fn forward(&self, times: Tensor<B, 1>) -> Tensor<B, 2> {
        let t = times.unsqueeze_dim::<2>(1);
        let rand_proj = self.proj.forward(t);
        (rand_proj * (2.0 * PI)).cos()
    }
}

// ---------------------------------------------------------------------------
// SingleConditioning  (Algorithm 21)
// ---------------------------------------------------------------------------

/// Conditions the score model on the trunk single representation + noise level.
#[derive(Module, Debug)]
pub struct SingleConditioning<B: Backend> {
    norm_single: LayerNorm<B>,
    single_embed: Linear<B>,
    fourier_embed: FourierEmbedding<B>,
    norm_fourier: LayerNorm<B>,
    fourier_to_single: Linear<B>,
    transitions: Vec<Transition<B>>,
}

impl<B: Backend> SingleConditioning<B> {
    pub fn new(
        device: &Device<B>,
        _sigma_data: f64,
        token_s: usize,
        dim_fourier: usize,
        num_transitions: usize,
        transition_expansion_factor: usize,
    ) -> Self {
        let two_ts = 2 * token_s;
        let norm_single = layer_norm_1d(device, two_ts);
        let single_embed = LinearConfig::new(two_ts, two_ts).init(device);
        let fourier_embed = FourierEmbedding::new(device, dim_fourier);
        let norm_fourier = layer_norm_1d(device, dim_fourier);
        let fourier_to_single = linear_no_bias(device, dim_fourier, two_ts);

        let mut transitions = Vec::new();
        for _ in 0..num_transitions {
            transitions.push(Transition::new(
                device,
                two_ts,
                Some(transition_expansion_factor * two_ts),
                None,
            ));
        }

        Self {
            norm_single,
            single_embed,
            fourier_embed,
            norm_fourier,
            fourier_to_single,
            transitions,
        }
    }

    /// Returns `(s_conditioned, normed_fourier)`.
    pub fn forward(
        &self,
        times: Tensor<B, 1>,
        s_trunk: Tensor<B, 3>,
        s_inputs: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let s = Tensor::cat(vec![s_trunk, s_inputs], 2);
        let mut s = self.single_embed.forward(self.norm_single.forward(s));

        let fourier_embed = self.fourier_embed.forward(times);
        let normed_fourier = self.norm_fourier.forward(fourier_embed);
        let fourier_to_single = self.fourier_to_single.forward(normed_fourier.clone());
        s = fourier_to_single.unsqueeze_dim::<3>(1) + s;

        for transition in &self.transitions {
            let t = transition.forward(s.clone(), None);
            s = t + s;
        }
        (s, normed_fourier)
    }
}

// ---------------------------------------------------------------------------
// PairwiseConditioning  (Algorithm 21)
// ---------------------------------------------------------------------------

/// Conditions the pairwise representation for diffusion.
#[derive(Module, Debug)]
pub struct PairwiseConditioning<B: Backend> {
    dim_pairwise_init_proj_norm: LayerNorm<B>,
    dim_pairwise_init_proj_linear: Linear<B>,
    transitions: Vec<Transition<B>>,
}

impl<B: Backend> PairwiseConditioning<B> {
    pub fn new(
        device: &Device<B>,
        token_z: usize,
        dim_token_rel_pos_feats: usize,
        num_transitions: usize,
        transition_expansion_factor: usize,
    ) -> Self {
        let combined = token_z + dim_token_rel_pos_feats;
        let dim_pairwise_init_proj_norm = layer_norm_1d(device, combined);
        let dim_pairwise_init_proj_linear = linear_no_bias(device, combined, token_z);

        let mut transitions = Vec::new();
        for _ in 0..num_transitions {
            transitions.push(Transition::new(
                device,
                token_z,
                Some(transition_expansion_factor * token_z),
                None,
            ));
        }

        Self {
            dim_pairwise_init_proj_norm,
            dim_pairwise_init_proj_linear,
            transitions,
        }
    }

    /// `z_trunk [B, N, N, tz]` + `rel_pos [B, N, N, tz]` → conditioned `z`.
    pub fn forward(
        &self,
        z_trunk: Tensor<B, 4>,
        token_rel_pos_feats: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let z = Tensor::cat(vec![z_trunk, token_rel_pos_feats], 3);
        let mut z = self
            .dim_pairwise_init_proj_linear
            .forward(self.dim_pairwise_init_proj_norm.forward(z));

        for transition in &self.transitions {
            let t = transition.forward(z.clone(), None);
            z = t + z;
        }
        z
    }
}

fn align_z_to_p_with_p<B: Backend>(p: &Tensor<B, 5>, z_to_p_out: Tensor<B, 5>) -> Tensor<B, 5> {
    let ps = p.dims();
    let zs = z_to_p_out.dims();
    if ps.len() == 5 && zs.len() == 5 && ps[0] == zs[0] && ps[4] == zs[4] {
        if ps[2] == zs[2] && ps[3] == zs[3] {
            return z_to_p_out;
        }
        if ps[2] == zs[3] && ps[3] == zs[2] {
            return z_to_p_out.swap_dims(2, 3);
        }
    }
    z_to_p_out
}

// ---------------------------------------------------------------------------
// AtomEncoder — flags / extra feats
// ---------------------------------------------------------------------------

/// Hyperparameters controlling which atom feature tensors are concatenated before `embed_atom_features`.
#[derive(Clone, Debug)]
pub struct AtomEncoderFlags {
    pub num_elements: usize,
    pub use_no_atom_char: bool,
    pub use_atom_backbone_feat: bool,
    pub use_residue_feats_atoms: bool,
    pub backbone_feat_dim: usize,
    pub num_tokens: usize,
}

impl Default for AtomEncoderFlags {
    fn default() -> Self {
        Self {
            num_elements: 128,
            use_no_atom_char: true,
            use_atom_backbone_feat: false,
            use_residue_feats_atoms: false,
            backbone_feat_dim: 17,
            num_tokens: 33,
        }
    }
}

impl AtomEncoderFlags {
    #[must_use]
    pub fn expected_atom_feature_dim(&self) -> usize {
        let mut d = 3 + 1 + self.num_elements;
        if !self.use_no_atom_char {
            d += 4 * 64;
        }
        if self.use_atom_backbone_feat {
            d += self.backbone_feat_dim;
        }
        if self.use_residue_feats_atoms {
            d += self.num_tokens + 1 + 4;
        }
        d
    }
}

/// Optional per-forward tensors for extended atom encodings.
pub struct AtomEncoderBatchFeats<'a, B: Backend> {
    pub ref_atom_name_chars: Option<&'a Tensor<B, 4, Int>>,
    pub atom_backbone_feat: Option<&'a Tensor<B, 3>>,
    pub res_type: Option<&'a Tensor<B, 3>>,
    pub modified: Option<&'a Tensor<B, 2>>,
    pub mol_type: Option<&'a Tensor<B, 2, Int>>,
}

// ---------------------------------------------------------------------------
// AtomEncoder
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct AtomEncoder<B: Backend> {
    embed_atom_features: Linear<B>,
    embed_atompair_ref_pos: Linear<B>,
    embed_atompair_ref_dist: Linear<B>,
    embed_atompair_mask: Linear<B>,
    #[module(skip)]
    atoms_per_window_queries: usize,
    #[module(skip)]
    atoms_per_window_keys: usize,
    #[module(skip)]
    structure_prediction: bool,
    #[module(skip)]
    flags: AtomEncoderFlags,
    s_to_c_trans_norm: Option<LayerNorm<B>>,
    s_to_c_trans_linear: Option<Linear<B>>,
    z_to_p_trans_norm: Option<LayerNorm<B>>,
    z_to_p_trans_linear: Option<Linear<B>>,
    c_to_p_trans_k: Linear<B>,
    c_to_p_trans_q: Linear<B>,
    p_mlp_1: Linear<B>,
    p_mlp_3: Linear<B>,
    p_mlp_5: Linear<B>,
    #[module(skip)]
    atom_s: usize,
    #[module(skip)]
    atom_z: usize,
}

impl<B: Backend> AtomEncoder<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device<B>,
        atom_s: usize,
        atom_z: usize,
        token_s: usize,
        token_z: usize,
        atoms_per_window_queries: usize,
        atoms_per_window_keys: usize,
        atom_feature_dim: usize,
        structure_prediction: bool,
        flags: AtomEncoderFlags,
    ) -> Self {
        let expected = flags.expected_atom_feature_dim();
        assert_eq!(
            atom_feature_dim, expected,
            "atom_feature_dim {atom_feature_dim} != expected {expected} for AtomEncoderFlags {flags:?}",
        );
        let embed_atom_features = LinearConfig::new(atom_feature_dim, atom_s).init(device);
        let embed_atompair_ref_pos = linear_no_bias(device, 3, atom_z);
        let embed_atompair_ref_dist = linear_no_bias(device, 1, atom_z);
        let embed_atompair_mask = linear_no_bias(device, 1, atom_z);

        let (s_to_c_trans_norm, s_to_c_trans_linear, z_to_p_trans_norm, z_to_p_trans_linear) =
            if structure_prediction {
                (
                    Some(layer_norm_1d(device, token_s)),
                    Some(linear_no_bias(device, token_s, atom_s)),
                    Some(layer_norm_1d(device, token_z)),
                    Some(linear_no_bias(device, token_z, atom_z)),
                )
            } else {
                (None, None, None, None)
            };

        let c_to_p_trans_k = linear_no_bias(device, atom_s, atom_z);
        let c_to_p_trans_q = linear_no_bias(device, atom_s, atom_z);
        let p_mlp_1 = linear_no_bias(device, atom_z, atom_z);
        let p_mlp_3 = linear_no_bias(device, atom_z, atom_z);
        let p_mlp_5 = linear_no_bias(device, atom_z, atom_z);

        Self {
            embed_atom_features,
            embed_atompair_ref_pos,
            embed_atompair_ref_dist,
            embed_atompair_mask,
            atoms_per_window_queries,
            atoms_per_window_keys,
            structure_prediction,
            flags,
            s_to_c_trans_norm,
            s_to_c_trans_linear,
            z_to_p_trans_norm,
            z_to_p_trans_linear,
            c_to_p_trans_k,
            c_to_p_trans_q,
            p_mlp_1,
            p_mlp_3,
            p_mlp_5,
            atom_s,
            atom_z,
        }
    }

    pub fn flags(&self) -> &AtomEncoderFlags {
        &self.flags
    }

    fn concat_atom_feats(
        &self,
        ref_pos: &Tensor<B, 3>,
        ref_charge: &Tensor<B, 2>,
        ref_element: &Tensor<B, 3>,
        atom_to_token: &Tensor<B, 3>,
        batch: Option<&AtomEncoderBatchFeats<'_, B>>,
    ) -> Tensor<B, 3> {
        let f = &self.flags;
        debug_assert_eq!(
            ref_element.dims()[2],
            f.num_elements,
            "ref_element last dim must match AtomEncoderFlags::num_elements"
        );
        let mut pieces = vec![
            ref_pos.clone(),
            ref_charge.clone().unsqueeze_dim::<3>(2),
            ref_element.clone(),
        ];
        if !f.use_no_atom_char {
            let chars = batch
                .and_then(|b| b.ref_atom_name_chars)
                .expect("ref_atom_name_chars required when use_no_atom_char is false");
            let [_, _, _, width] = chars.dims();
            pieces.push(chars.clone().float().reshape([chars.dims()[0], chars.dims()[1], 4 * width]));
        }
        if f.use_atom_backbone_feat {
            let bb = batch
                .and_then(|b| b.atom_backbone_feat)
                .expect("atom_backbone_feat required when use_atom_backbone_feat");
            pieces.push(bb.clone());
        }
        if f.use_residue_feats_atoms {
            let b = batch.expect("batch required when use_residue_feats_atoms");
            let res_type = b.res_type.expect("res_type required when use_residue_feats_atoms");
            let modified = b.modified.expect("modified required when use_residue_feats_atoms");
            let mol_type = b.mol_type.expect("mol_type required when use_residue_feats_atoms");
            let mol_oh = one_hot_2d(mol_type.clone(), 4);
            let res_feats = Tensor::cat(
                vec![
                    res_type.clone(),
                    modified.clone().unsqueeze_dim::<3>(2),
                    mol_oh,
                ],
                2,
            );
            let atom_res = atom_to_token.clone().matmul(res_feats);
            pieces.push(atom_res);
        }
        Tensor::cat(pieces, 2)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        ref_pos: Tensor<B, 3>,
        ref_charge: Tensor<B, 2>,
        ref_element: Tensor<B, 3>,
        atom_pad_mask: Tensor<B, 2>,
        ref_space_uid: Tensor<B, 2, Int>,
        atom_to_token: Tensor<B, 3>,
        s_trunk: Option<Tensor<B, 3>>,
        z: Option<Tensor<B, 4>>,
        batch: Option<&AtomEncoderBatchFeats<'_, B>>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 5>, Tensor<B, 2>) {
        let [b, n, _] = ref_pos.dims();
        let w = self.atoms_per_window_queries;
        let h = self.atoms_per_window_keys;
        let k = n / w;

        let atom_feats = self.concat_atom_feats(
            &ref_pos,
            &ref_charge,
            &ref_element,
            &atom_to_token,
            batch,
        );
        let c = self.embed_atom_features.forward(atom_feats);

        let indexing_matrix = get_indexing_matrix::<B>(k, w, h, &ref_pos.device());

        let atom_ref_pos_queries = ref_pos.clone().reshape([b, k, w, 1, 3]);
        let atom_ref_pos_keys = single_to_keys(ref_pos.clone(), &indexing_matrix, w, h)
            .reshape([b, k, 1, h, 3]);

        let d = atom_ref_pos_keys - atom_ref_pos_queries;
        let d_norm: Tensor<B, 4> =
            (d.clone().powi_scalar(2).sum_dim(4).squeeze_dim::<4>(4) + 1.0).recip();

        let atom_mask = atom_pad_mask.clone().bool();
        let atom_mask_queries = atom_mask.clone().reshape([b, k, w, 1]);
        let atom_mask_keys = single_to_keys(
            atom_pad_mask.clone().unsqueeze_dim::<3>(2),
            &indexing_matrix,
            w,
            h,
        )
        .reshape([b, k, 1, h])
        .bool();
        let atom_uid_queries = ref_space_uid.clone().reshape([b, k, w, 1]);
        let atom_uid_keys = single_to_keys(
            ref_space_uid.clone().float().unsqueeze_dim::<3>(2),
            &indexing_matrix,
            w,
            h,
        )
        .reshape([b, k, 1, h])
        .int();

        let v = (atom_mask_queries
            .bool_and(atom_mask_keys)
            .bool_and(atom_uid_queries.equal(atom_uid_keys)))
        .float()
        .unsqueeze_dim::<5>(4);

        let d_norm_in = d_norm.unsqueeze_dim::<5>(4);
        let mut p = self.embed_atompair_ref_pos.forward(d) * v.clone();
        p = p + self.embed_atompair_ref_dist.forward(d_norm_in) * v.clone();
        p = p + self.embed_atompair_mask.forward(v.clone()) * v;

        let q = c.clone();

        let mut c = c;
        if self.structure_prediction {
            if let (Some(s_trunk), Some(z)) = (s_trunk, z) {
                let s_to_c = self
                    .s_to_c_trans_linear
                    .as_ref()
                    .expect("AtomEncoder: s_to_c_trans_linear missing despite structure_prediction")
                    .forward(
                        self.s_to_c_trans_norm
                            .as_ref()
                            .expect("AtomEncoder: s_to_c_trans_norm missing despite structure_prediction")
                            .forward(s_trunk),
                    );
                let s_to_c = atom_to_token.clone().matmul(s_to_c);
                c = c + s_to_c;

                let z_to_p = self
                    .z_to_p_trans_linear
                    .as_ref()
                    .expect("AtomEncoder: z_to_p_trans_linear missing despite structure_prediction")
                    .forward(
                        self.z_to_p_trans_norm
                            .as_ref()
                            .expect("AtomEncoder: z_to_p_trans_norm missing despite structure_prediction")
                            .forward(z),
                    );
                let atom_to_token_queries = atom_to_token.clone().reshape([b, k, w, atom_to_token.dims()[2]]);
                let atom_to_token_keys =
                    single_to_keys(atom_to_token.clone(), &indexing_matrix, w, h);
                let z_to_p_out = einsum_bijd_bwki_bwlj_bwkld(
                    z_to_p,
                    atom_to_token_queries,
                    atom_to_token_keys,
                );
                let z_to_p_out = align_z_to_p_with_p(&p, z_to_p_out);
                if p.dims() != z_to_p_out.dims() {
                    panic!(
                        "AtomEncoder: z_to_p_out shape {:?} does not match p {:?} after W/H alignment",
                        z_to_p_out.dims(),
                        p.dims()
                    );
                }
                p = p + z_to_p_out;
            }
        }

        let c_q = c.clone().reshape([b, k, w, 1, c.dims()[2]]);
        let c_keys = single_to_keys(c.clone(), &indexing_matrix, w, h)
            .reshape([b, k, 1, h, c.dims()[2]]);
        p = p + self.c_to_p_trans_q.forward(relu(c_q));
        p = p + self.c_to_p_trans_k.forward(relu(c_keys));

        let p_mlp = self.p_mlp_5.forward(relu(self.p_mlp_3.forward(relu(self.p_mlp_1.forward(relu(p.clone()))))));
        p = p + p_mlp;

        (q, c, p, indexing_matrix)
    }
}

// ---------------------------------------------------------------------------
// AtomAttentionEncoder
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct AtomAttentionEncoder<B: Backend> {
    #[module(skip)]
    structure_prediction: bool,
    r_to_q_trans: Option<Linear<B>>,
    atom_encoder: AtomTransformer<B>,
    atom_to_token_trans_linear: Linear<B>,
    #[module(skip)]
    token_s_out: usize,
}

impl<B: Backend> AtomAttentionEncoder<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device<B>,
        atom_s: usize,
        token_s: usize,
        atoms_per_window_queries: usize,
        atoms_per_window_keys: usize,
        atom_encoder_depth: usize,
        atom_encoder_heads: usize,
        structure_prediction: bool,
    ) -> Self {
        let r_to_q_trans = if structure_prediction {
            Some(linear_no_bias(device, 3, atom_s))
        } else {
            None
        };

        let atom_encoder = AtomTransformer::new(
            device,
            atoms_per_window_queries,
            atoms_per_window_keys,
            atom_encoder_depth,
            atom_encoder_heads,
            atom_s,
            Some(atom_s),
        );

        let token_s_out = if structure_prediction { 2 * token_s } else { token_s };
        let atom_to_token_trans_linear = linear_no_bias(device, atom_s, token_s_out);

        Self {
            structure_prediction,
            r_to_q_trans,
            atom_encoder,
            atom_to_token_trans_linear,
            token_s_out,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        q: Tensor<B, 3>,
        c: Tensor<B, 3>,
        atom_enc_bias: Tensor<B, 5>,
        atom_pad_mask: Tensor<B, 2>,
        atom_to_token: Tensor<B, 3>,
        r: Tensor<B, 3>,
        multiplicity: usize,
        indexing_matrix: &Tensor<B, 2>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let mut q = if self.structure_prediction {
            let q_exp = repeat_interleave_dim0(q, multiplicity);
            let r_to_q = self
                .r_to_q_trans
                .as_ref()
                .expect("AtomDecoder: r_to_q_trans missing despite structure_prediction")
                .forward(r);
            q_exp + r_to_q
        } else {
            q
        };

        let c = repeat_interleave_dim0(c, multiplicity);
        let mask = repeat_interleave_dim0(atom_pad_mask, multiplicity);

        q = self.atom_encoder.forward(
            q,
            c.clone(),
            atom_enc_bias,
            mask,
            multiplicity,
            indexing_matrix,
        );

        let q_skip = q.clone();
        let c_skip = c.clone();

        let q_to_a = relu(self.atom_to_token_trans_linear.forward(q));
        let a2t = repeat_interleave_dim0(atom_to_token, multiplicity);
        let a2t_mean = a2t.clone() / (a2t.sum_dim(1) + 1e-6);
        let a = a2t_mean.swap_dims(1, 2).matmul(q_to_a);

        (a, q_skip, c_skip)
    }
}

// ---------------------------------------------------------------------------
// AtomAttentionDecoder
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct AtomAttentionDecoder<B: Backend> {
    a_to_q_trans: Linear<B>,
    atom_decoder: AtomTransformer<B>,
    atom_feat_to_atom_pos_update_norm: LayerNorm<B>,
    atom_feat_to_atom_pos_update_linear: Linear<B>,
    #[module(skip)]
    atom_s: usize,
}

impl<B: Backend> AtomAttentionDecoder<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device<B>,
        atom_s: usize,
        token_s: usize,
        attn_window_queries: usize,
        attn_window_keys: usize,
        atom_decoder_depth: usize,
        atom_decoder_heads: usize,
    ) -> Self {
        let a_to_q_trans = linear_no_bias(device, 2 * token_s, atom_s);
        let atom_decoder = AtomTransformer::new(
            device,
            attn_window_queries,
            attn_window_keys,
            atom_decoder_depth,
            atom_decoder_heads,
            atom_s,
            Some(atom_s),
        );
        let atom_feat_norm = layer_norm_1d(device, atom_s);
        let atom_feat_linear = linear_no_bias(device, atom_s, 3);

        Self {
            a_to_q_trans,
            atom_decoder,
            atom_feat_to_atom_pos_update_norm: atom_feat_norm,
            atom_feat_to_atom_pos_update_linear: atom_feat_linear,
            atom_s,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        a: Tensor<B, 3>,
        q: Tensor<B, 3>,
        c: Tensor<B, 3>,
        atom_dec_bias: Tensor<B, 5>,
        atom_pad_mask: Tensor<B, 2>,
        atom_to_token: Tensor<B, 3>,
        multiplicity: usize,
        indexing_matrix: &Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let a2t = repeat_interleave_dim0(atom_to_token, multiplicity);
        let a_to_q = self.a_to_q_trans.forward(a);
        let a_to_q = a2t.matmul(a_to_q);

        let mut q = q + a_to_q;
        let mask = repeat_interleave_dim0(atom_pad_mask, multiplicity);

        q = self.atom_decoder.forward(
            q,
            c,
            atom_dec_bias,
            mask,
            multiplicity,
            indexing_matrix,
        );

        self.atom_feat_to_atom_pos_update_linear
            .forward(self.atom_feat_to_atom_pos_update_norm.forward(q))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn align_z_to_p_with_p_fixes_swapped_window_axes() {
        let device = Default::default();
        let p = Tensor::<B, 5>::zeros([1, 2, 32, 128, 16], &device);
        let swapped = Tensor::<B, 5>::zeros([1, 2, 128, 32, 16], &device);
        let fixed = align_z_to_p_with_p(&p, swapped);
        assert_eq!(fixed.dims(), p.dims());
    }

    #[test]
    fn fourier_embedding_shape() {
        let device = Default::default();
        let fe = FourierEmbedding::<B>::new(&device, 256);
        let t = Tensor::<B, 1>::random([4], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let out = fe.forward(t);
        assert_eq!(out.dims(), [4, 256]);
    }

    #[test]
    fn single_conditioning_shape() {
        let device = Default::default();
        let ts = 32_usize;
        let sc = SingleConditioning::<B>::new(&device, 16.0, ts, 64, 2, 2);
        let times = Tensor::<B, 1>::random([2], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let s_trunk = Tensor::<B, 3>::random(
            [2, 8, ts],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let s_inputs = Tensor::<B, 3>::random(
            [2, 8, ts],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let (s, nf) = sc.forward(times, s_trunk, s_inputs);
        assert_eq!(s.dims(), [2, 8, 2 * ts]);
        assert_eq!(nf.dims(), [2, 64]);
    }

    #[test]
    fn pairwise_conditioning_shape() {
        let device = Default::default();
        let tz = 32_usize;
        let pc = PairwiseConditioning::<B>::new(&device, tz, tz, 2, 2);
        let z = Tensor::<B, 4>::random(
            [2, 8, 8, tz],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let rel = Tensor::<B, 4>::random(
            [2, 8, 8, tz],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let out = pc.forward(z, rel);
        assert_eq!(out.dims(), [2, 8, 8, tz]);
    }
}
