//! Boltz2 `MSAModule` (`modules/trunkv2.py`).

use burn::module::Module;
use burn::nn::Linear;
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Int, Tensor};

use crate::burn_compat::linear_no_bias;
use crate::layers::{
    OuterProductMeanMsa, PairWeightedAveraging, PairformerNoSeqLayer, Transition,
};
use crate::tensor_ops::one_hot_int;

use super::input_embedder::BOLTZ_NUM_TOKENS;

/// Collated MSA-related tensors (Boltz featurizer / collate contract).
pub struct MsaFeatures<'a, B: Backend> {
    pub msa: &'a Tensor<B, 3, Int>,
    pub msa_mask: &'a Tensor<B, 3>,
    pub has_deletion: &'a Tensor<B, 3, Int>,
    pub deletion_value: &'a Tensor<B, 3>,
    pub msa_paired: &'a Tensor<B, 3, Int>,
    pub token_pad_mask: &'a Tensor<B, 2>,
}

#[derive(Module, Debug)]
struct MsaLayerBlock<B: Backend> {
    pair_weighted_averaging: PairWeightedAveraging<B>,
    msa_transition: Transition<B>,
    outer_product_mean: OuterProductMeanMsa<B>,
    pairformer_layer: PairformerNoSeqLayer<B>,
    msa_dropout: f64,
}

impl<B: Backend> MsaLayerBlock<B> {
    fn new(
        device: &Device<B>,
        msa_s: usize,
        token_z: usize,
        msa_dropout: f64,
        z_dropout: f64,
        pairwise_head_width: usize,
        pairwise_num_heads: usize,
    ) -> Self {
        Self {
            pair_weighted_averaging: PairWeightedAveraging::new(
                device, msa_s, token_z, 32, 8, None,
            ),
            msa_transition: Transition::new(device, msa_s, Some(msa_s * 4), None),
            outer_product_mean: OuterProductMeanMsa::new(device, msa_s, 32, token_z),
            pairformer_layer: PairformerNoSeqLayer::new(
                device,
                token_z,
                z_dropout,
                pairwise_head_width,
                pairwise_num_heads,
            ),
            msa_dropout,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        z: Tensor<B, 4>,
        m: Tensor<B, 4>,
        pair_mask: Tensor<B, 3>,
        msa_mask_float: Tensor<B, 3>,
        training: bool,
        chunk_size_tri_attn: Option<i64>,
        use_kernels: bool,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let msa_drop = msa_dropout_mask(self.msa_dropout, &m, training);
        let pwa = self.pair_weighted_averaging.forward(m.clone(), z.clone(), pair_mask.clone());
        let m = {
            let m1 = m + msa_drop * pwa;
            let m2 = self.msa_transition.forward(m1.clone(), None);
            m1 + m2
        };

        let mut z = z + self.outer_product_mean.forward(m.clone(), msa_mask_float);
        z = self.pairformer_layer.forward(
            z,
            pair_mask,
            chunk_size_tri_attn,
            training,
            use_kernels,
        );
        (z, m)
    }
}

fn msa_dropout_mask<B: Backend>(
    dropout: f64,
    m: &Tensor<B, 4>,
    training: bool,
) -> Tensor<B, 4> {
    if !training || dropout == 0.0 {
        return Tensor::<B, 4>::ones([1, 1, 1, 1], &m.device());
    }
    let scale = 1.0 / (1.0 - dropout);
    let v = m.clone().slice([0..m.dims()[0], 0..m.dims()[1], 0..1, 0..1]);
    let thr = Tensor::<B, 4>::full([1, 1, 1, 1], dropout, &m.device());
    v.lower_equal(thr).float() * scale
}

#[derive(Module, Debug)]
pub struct MsaModule<B: Backend> {
    token_s: usize,
    token_z: usize,
    msa_s: usize,
    use_paired_feature: bool,
    s_proj: Linear<B>,
    msa_proj: Linear<B>,
    layers: Vec<MsaLayerBlock<B>>,
}

impl<B: Backend> MsaModule<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device<B>,
        token_s: usize,
        token_z: usize,
        msa_s: Option<usize>,
        msa_blocks: Option<usize>,
        msa_dropout: Option<f64>,
        z_dropout: Option<f64>,
        use_paired_feature: Option<bool>,
        pairwise_head_width: Option<usize>,
        pairwise_num_heads: Option<usize>,
    ) -> Self {
        let msa_s = msa_s.unwrap_or(64);
        let msa_blocks = msa_blocks.unwrap_or(4);
        let msa_dropout = msa_dropout.unwrap_or(0.0);
        let z_dropout = z_dropout.unwrap_or(0.0);
        let use_paired_feature = use_paired_feature.unwrap_or(true);
        let pairwise_head_width = pairwise_head_width.unwrap_or(32);
        let pairwise_num_heads = pairwise_num_heads.unwrap_or(4);
        let msa_in = BOLTZ_NUM_TOKENS + 2 + usize::from(use_paired_feature);

        let layers = (0..msa_blocks)
            .map(|_| {
                MsaLayerBlock::new(
                    device,
                    msa_s,
                    token_z,
                    msa_dropout,
                    z_dropout,
                    pairwise_head_width,
                    pairwise_num_heads,
                )
            })
            .collect();

        Self {
            token_s,
            token_z,
            msa_s,
            use_paired_feature,
            s_proj: linear_no_bias(device, token_s, msa_s),
            msa_proj: linear_no_bias(device, msa_in, msa_s),
            layers,
        }
    }

    pub fn forward_trunk_step(
        &self,
        z: Tensor<B, 4>,
        s: Tensor<B, 3>,
        feats: Option<&MsaFeatures<'_, B>>,
        training: bool,
        chunk_size_tri_attn: Option<i64>,
        use_kernels: bool,
    ) -> Tensor<B, 4> {
        let Some(feats) = feats else {
            return z;
        };

        let msa_oh = one_hot_int(feats.msa.clone(), BOLTZ_NUM_TOKENS);
        let hd = feats
            .has_deletion
            .clone()
            .float()
            .unsqueeze_dim::<4>(3);
        let dv = feats.deletion_value.clone().unsqueeze_dim::<4>(3);
        let mut pieces = vec![msa_oh, hd, dv];
        if self.use_paired_feature {
            pieces.push(feats.msa_paired.clone().float().unsqueeze_dim::<4>(3));
        }
        let m_cat = Tensor::cat(pieces, 3);
        let m_lin = self.msa_proj.forward(m_cat);
        let s_lin = self.s_proj.forward(s).unsqueeze_dim::<4>(1);
        let mut m = m_lin + s_lin;

        let msa_mask_f = feats.msa_mask.clone();
        let tm = feats.token_pad_mask.clone();
        let pair_mask = tm.clone().unsqueeze_dim::<3>(2) * tm.unsqueeze_dim::<3>(1);

        let mut z = z;
        for layer in &self.layers {
            let (zn, mn) = layer.forward(
                z,
                m,
                pair_mask.clone(),
                msa_mask_f.clone(),
                training,
                chunk_size_tri_attn,
                use_kernels,
            );
            z = zn;
            m = mn;
        }
        z
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn msa_module_forward_shapes() {
        let device = Default::default();
        let m = MsaModule::<B>::new(&device, 32, 24, Some(16), Some(2), Some(0.0), Some(0.0), Some(true), None, None);
        let z = Tensor::<B, 4>::random(
            [1, 6, 6, 24],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let emb = Tensor::<B, 3>::random(
            [1, 6, 32],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let msa = Tensor::<B, 3, Int>::zeros([1, 4, 6], &device);
        let msa_mask = Tensor::<B, 3>::ones([1, 4, 6], &device);
        let has_deletion = Tensor::<B, 3, Int>::zeros([1, 4, 6], &device);
        let deletion_value = Tensor::<B, 3>::zeros([1, 4, 6], &device);
        let msa_paired = Tensor::<B, 3, Int>::zeros([1, 4, 6], &device);
        let token_pad_mask = Tensor::<B, 2>::ones([1, 6], &device);
        let feats = MsaFeatures {
            msa: &msa,
            msa_mask: &msa_mask,
            has_deletion: &has_deletion,
            deletion_value: &deletion_value,
            msa_paired: &msa_paired,
            token_pad_mask: &token_pad_mask,
        };
        let out = m.forward_trunk_step(z, emb, Some(&feats), false, None, false);
        assert_eq!(out.dims(), [1, 6, 6, 24]);
    }

    #[test]
    fn msa_module_none_feats_identity() {
        let device = Default::default();
        let m = MsaModule::<B>::new(&device, 32, 16, None, Some(1), None, None, None, None, None);
        let z = Tensor::<B, 4>::ones([1, 3, 3, 16], &device);
        let s = Tensor::<B, 3>::zeros([1, 3, 32], &device);
        let out = m.forward_trunk_step(z.clone(), s, None, false, None, false);
        let diff = (out - z).abs().max();
        assert!(diff.into_scalar() < 1e-6);
    }
}
