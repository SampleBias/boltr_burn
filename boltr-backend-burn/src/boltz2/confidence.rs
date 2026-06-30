//! Confidence head — `boltz.model.modules.confidencev2` (`ConfidenceModule` + `ConfidenceHeads`).
//!
//! Python reference: `boltz-reference/src/boltz/model/modules/confidencev2.py`.

use std::collections::BTreeMap;

use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, Linear};
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Int, Tensor};

use crate::burn_compat::{layer_norm_1d, linear_no_bias};
use crate::layers::PairformerModule;
use crate::tensor_ops::{
    cdist_euclidean, dist_to_bin_indices, linspace, repeat_interleave_dim0,
    repeat_interleave_dim0_int,
};

use super::confidence_utils::{
    compute_aggregated_metric, compute_aggregated_metric_token, compute_ptms,
    CHAIN_TYPE_NONPOLYMER,
};

/// Configuration for [`ConfidenceModule`] (subset of Python kwargs + pairformer sizing).
#[derive(Debug, Clone)]
pub struct ConfidenceModuleConfig {
    pub num_dist_bins: usize,
    pub max_dist: f64,
    pub pairformer_num_blocks: usize,
    pub pairformer_num_heads: Option<usize>,
    pub no_update_s: bool,
    pub token_level_confidence: bool,
    pub num_plddt_bins: usize,
    pub num_pde_bins: usize,
    pub num_pae_bins: usize,
    pub use_separate_heads: bool,
}

impl Default for ConfidenceModuleConfig {
    fn default() -> Self {
        Self {
            num_dist_bins: 64,
            max_dist: 22.0,
            pairformer_num_blocks: 4,
            pairformer_num_heads: Some(16),
            no_update_s: false,
            token_level_confidence: true,
            num_plddt_bins: 50,
            num_pde_bins: 64,
            num_pae_bins: 64,
            use_separate_heads: false,
        }
    }
}

fn embed_dist_bins<B: Backend>(
    embed: &Embedding<B>,
    bin_indices: Tensor<B, 3, Int>,
) -> Tensor<B, 4> {
    let [b, n, n2] = bin_indices.dims();
    let token_z = embed.weight.dims()[1];
    let flat = bin_indices.reshape([b, n * n2]);
    embed.forward(flat).reshape([b, n, n2, token_z])
}

/// Full confidence stack + heads (`confidence_module` in Lightning).
#[derive(Module, Debug)]
pub struct ConfidenceModule<B: Backend> {
    boundaries: Vec<f64>,
    dist_bin_pairwise_embed: Embedding<B>,
    s_to_z: Linear<B>,
    s_to_z_transpose: Linear<B>,
    s_inputs_norm: LayerNorm<B>,
    s_norm: Option<LayerNorm<B>>,
    z_norm: LayerNorm<B>,
    pairformer_stack: PairformerModule<B>,
    heads: ConfidenceHeads<B>,
    token_level_confidence: bool,
}

impl<B: Backend> ConfidenceModule<B> {
    pub fn new(
        device: &Device<B>,
        token_s: usize,
        token_z: usize,
        cfg: &ConfidenceModuleConfig,
    ) -> Self {
        if cfg.use_separate_heads {
            panic!("use_separate_heads=true not implemented in Rust port yet");
        }

        let num_dist = cfg.num_dist_bins;
        let boundaries = linspace(2.0, cfg.max_dist, num_dist.saturating_sub(1));
        let num_heads = cfg.pairformer_num_heads.unwrap_or(16);

        Self {
            boundaries,
            dist_bin_pairwise_embed: EmbeddingConfig::new(num_dist, token_z).init(device),
            s_to_z: linear_no_bias(device, token_s, token_z),
            s_to_z_transpose: linear_no_bias(device, token_s, token_z),
            s_inputs_norm: layer_norm_1d(device, token_s),
            s_norm: if cfg.no_update_s {
                None
            } else {
                Some(layer_norm_1d(device, token_s))
            },
            z_norm: layer_norm_1d(device, token_z),
            pairformer_stack: PairformerModule::new(
                device,
                token_s,
                token_z,
                cfg.pairformer_num_blocks,
                num_heads,
                0.25,
            ),
            heads: ConfidenceHeads::new(device, token_s, token_z, cfg),
            token_level_confidence: cfg.token_level_confidence,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        s_inputs: Tensor<B, 3>,
        s: Tensor<B, 3>,
        z: Tensor<B, 4>,
        x_pred: Tensor<B, 3>,
        token_pad_mask: Tensor<B, 2>,
        asym_id: Tensor<B, 2, Int>,
        mol_type: Tensor<B, 2, Int>,
        token_to_rep_atom: Tensor<B, 3>,
        frames_idx: Tensor<B, 3, Int>,
        pred_distogram_logits: Tensor<B, 4>,
        multiplicity: usize,
    ) -> ConfidenceOutput<B> {
        let _ = self.token_level_confidence;

        let s_inputs = self.s_inputs_norm.forward(s_inputs);
        let mut s = s;
        if let Some(sn) = &self.s_norm {
            s = sn.forward(s);
        }
        let z = self.z_norm.forward(z);

        let mut z = z
            + self.s_to_z.forward(s_inputs.clone()).unsqueeze_dim::<4>(2)
            + self
                .s_to_z_transpose
                .forward(s_inputs.clone())
                .unsqueeze_dim::<4>(1);

        let s = repeat_interleave_dim0(s, multiplicity);
        z = repeat_interleave_dim0(z, multiplicity);
        let _s_inputs = repeat_interleave_dim0(s_inputs, multiplicity);
        let token_to_rep_atom = repeat_interleave_dim0(token_to_rep_atom, multiplicity);

        let x_pred_repr = token_to_rep_atom.matmul(x_pred.clone());
        let d = cdist_euclidean(x_pred_repr);
        let distogram = dist_to_bin_indices(d.clone(), &self.boundaries);
        let distogram = embed_dist_bins(&self.dist_bin_pairwise_embed, distogram);
        let z = z + distogram;

        let mask = repeat_interleave_dim0(token_pad_mask.clone(), multiplicity);
        let pair_mask = mask.clone().unsqueeze_dim::<3>(2) * mask.clone().unsqueeze_dim::<3>(1);

        let (s, z) = self
            .pairformer_stack
            .forward(s, z, pair_mask.clone(), pair_mask, None, false, false);

        self.heads.forward(
            s,
            z,
            x_pred,
            d,
            token_pad_mask,
            asym_id,
            mol_type,
            pred_distogram_logits,
            multiplicity,
            frames_idx,
        )
    }
}

/// Head outputs (logits + derived scalars). Mirrors Python dict keys used in `boltz2.py`.
pub struct ConfidenceOutput<B: Backend> {
    pub pae_logits: Tensor<B, 4>,
    pub pde_logits: Tensor<B, 4>,
    pub plddt_logits: Tensor<B, 3>,
    pub resolved_logits: Tensor<B, 3>,
    pub pae: Tensor<B, 3>,
    pub pde: Tensor<B, 3>,
    pub plddt: Tensor<B, 2>,
    pub complex_plddt: Tensor<B, 1>,
    pub complex_iplddt: Tensor<B, 1>,
    pub complex_pde: Tensor<B, 1>,
    pub complex_ipde: Tensor<B, 1>,
    pub ptm: Tensor<B, 1>,
    pub iptm: Tensor<B, 1>,
    pub ligand_iptm: Tensor<B, 1>,
    pub protein_iptm: Tensor<B, 1>,
    pub pair_chains_iptm: BTreeMap<i64, BTreeMap<i64, Tensor<B, 1>>>,
}

#[derive(Module, Debug)]
struct ConfidenceHeads<B: Backend> {
    to_pae_logits: Linear<B>,
    to_pde_logits: Linear<B>,
    to_plddt_logits: Linear<B>,
    to_resolved_logits: Linear<B>,
    token_level_confidence: bool,
}

impl<B: Backend> ConfidenceHeads<B> {
    fn new(
        device: &Device<B>,
        token_s: usize,
        token_z: usize,
        cfg: &ConfidenceModuleConfig,
    ) -> Self {
        Self {
            to_pae_logits: linear_no_bias(device, token_z, cfg.num_pae_bins),
            to_pde_logits: linear_no_bias(device, token_z, cfg.num_pde_bins),
            to_plddt_logits: linear_no_bias(device, token_s, cfg.num_plddt_bins),
            to_resolved_logits: linear_no_bias(device, token_s, 2),
            token_level_confidence: cfg.token_level_confidence,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        s: Tensor<B, 3>,
        z: Tensor<B, 4>,
        x_pred: Tensor<B, 3>,
        d: Tensor<B, 3>,
        token_pad_mask: Tensor<B, 2>,
        asym_id: Tensor<B, 2, Int>,
        mol_type: Tensor<B, 2, Int>,
        pred_distogram_logits: Tensor<B, 4>,
        multiplicity: usize,
        frames_idx: Tensor<B, 3, Int>,
    ) -> ConfidenceOutput<B> {
        assert!(
            self.token_level_confidence,
            "atom-level confidence path not ported"
        );

        let pae_logits = self.to_pae_logits.forward(z.clone());
        let pde_logits = self.to_pde_logits.forward(z.clone() + z.clone().swap_dims(1, 2));
        let resolved_logits = self.to_resolved_logits.forward(s.clone());
        let plddt_logits = self.to_plddt_logits.forward(s);

        let pde = compute_aggregated_metric(pde_logits.clone(), 32.0);
        let pred_log = repeat_interleave_dim0(pred_distogram_logits, multiplicity);
        let pred_distogram_prob = softmax(pred_log, 3);

        let nb = pred_distogram_prob.dims()[3];
        let n_contact_bins = 20_usize.min(nb);
        let device = pred_distogram_prob.device();
        let contacts = if n_contact_bins == nb {
            Tensor::<B, 4>::ones([1, 1, 1, nb], &device)
        } else {
            Tensor::cat(
                vec![
                    Tensor::<B, 4>::ones([1, 1, 1, n_contact_bins], &device),
                    Tensor::<B, 4>::zeros([1, 1, 1, nb - n_contact_bins], &device),
                ],
                3,
            )
        };
        let prob_contact = (pred_distogram_prob * contacts).sum_dim(3).squeeze_dim::<3>(3);

        let token_pad_mask_m = repeat_interleave_dim0(token_pad_mask.clone(), multiplicity);
        let n = token_pad_mask_m.dims()[1];
        let eye = Tensor::<B, 2>::eye(n, &device).unsqueeze_dim::<3>(0);
        let token_pad_pair_mask = token_pad_mask_m.clone().unsqueeze_dim::<3>(2)
            * token_pad_mask_m.clone().unsqueeze_dim::<3>(1)
            * (Tensor::<B, 3>::ones([1, n, n], &device) - eye);
        let token_pair_mask = token_pad_pair_mask * prob_contact;

        let plddt = compute_aggregated_metric_token(plddt_logits.clone(), 1.0);
        let complex_plddt = (plddt.clone() * token_pad_mask_m.clone()).sum_dim(1).squeeze_dim::<1>(1)
            / token_pad_mask_m.clone().sum_dim(1).squeeze_dim::<1>(1);

        let is_contact = d.lower_elem(8.0).float();
        let is_different_chain = repeat_interleave_dim0_int(asym_id.clone(), multiplicity)
            .unsqueeze_dim::<3>(2)
            .not_equal(
                repeat_interleave_dim0_int(asym_id.clone(), multiplicity).unsqueeze_dim::<3>(1),
            )
            .float();
        let is_ligand_token = repeat_interleave_dim0_int(mol_type.clone(), multiplicity)
            .equal_elem(CHAIN_TYPE_NONPOLYMER)
            .float();

        let token_interface_mask = (is_contact
            * is_different_chain.clone()
            * (Tensor::<B, 2>::ones(is_ligand_token.dims(), &device) - is_ligand_token.clone())
                .unsqueeze_dim::<3>(2))
            .max_dim(2)
            .squeeze_dim::<2>(2);
        let token_non_interface_mask = (Tensor::<B, 2>::ones(token_interface_mask.dims(), &device)
            - token_interface_mask.clone())
            * (Tensor::<B, 2>::ones(is_ligand_token.dims(), &device) - is_ligand_token.clone());

        let ligand_weight = 20.0;
        let non_interface_weight = 1.0;
        let interface_weight = 10.0;
        let iplddt_weight = is_ligand_token * ligand_weight
            + token_interface_mask * interface_weight
            + token_non_interface_mask * non_interface_weight;
        let complex_iplddt = (plddt.clone() * token_pad_mask_m.clone() * iplddt_weight.clone())
            .sum_dim(1)
            .squeeze_dim::<1>(1)
            / (token_pad_mask_m.clone() * iplddt_weight).sum_dim(1).squeeze_dim::<1>(1);

        let pae = compute_aggregated_metric(pae_logits.clone(), 32.0);

        let asym_m = repeat_interleave_dim0_int(asym_id.clone(), multiplicity);
        let token_interface_pair_mask = token_pair_mask.clone()
            * asym_m
                .clone()
                .unsqueeze_dim::<3>(2)
                .not_equal(asym_m.unsqueeze_dim::<3>(1))
                .float();
        let complex_pde = (pde.clone() * token_pair_mask.clone()).sum_dims_squeeze::<1, _>(&[1, 2])
            / token_pair_mask.sum_dims_squeeze::<1, _>(&[1, 2]);
        let complex_ipde = (pde.clone() * token_interface_pair_mask.clone()).sum_dims_squeeze::<1, _>(&[1, 2])
            / (token_interface_pair_mask.sum_dims_squeeze::<1, _>(&[1, 2]) + 1e-5);

        let (ptm, iptm, ligand_iptm, protein_iptm, pair_chains_iptm) = compute_ptms(
            pae_logits.clone(),
            x_pred,
            frames_idx,
            asym_id.clone(),
            mol_type,
            token_pad_mask,
            multiplicity,
        );

        ConfidenceOutput {
            pae_logits,
            pde_logits,
            plddt_logits,
            resolved_logits,
            pae,
            pde: pde.clone(),
            plddt,
            complex_plddt,
            complex_iplddt,
            complex_pde,
            complex_ipde,
            ptm,
            iptm,
            ligand_iptm,
            protein_iptm,
            pair_chains_iptm,
        }
    }
}

pub type ConfidenceV2<B> = ConfidenceModule<B>;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn confidence_module_forward_smoke() {
        let device = Default::default();
        let token_s = 32;
        let token_z = 16;
        let n = 8;
        let a = 8;
        let cfg = ConfidenceModuleConfig {
            num_dist_bins: 32,
            max_dist: 22.0,
            pairformer_num_blocks: 1,
            pairformer_num_heads: Some(4),
            no_update_s: false,
            token_level_confidence: true,
            num_plddt_bins: 16,
            num_pde_bins: 16,
            num_pae_bins: 16,
            use_separate_heads: false,
        };

        let m = ConfidenceModule::<B>::new(&device, token_s, token_z, &cfg);

        let b = 1;
        let s_inputs = Tensor::<B, 3>::random(
            [b, n, token_s],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let s = Tensor::<B, 3>::random(
            [b, n, token_s],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let z = Tensor::<B, 4>::random(
            [b, n, n, token_z],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let x_pred = Tensor::<B, 3>::random(
            [b, a, 3],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let token_pad_mask = Tensor::<B, 2>::ones([b, n], &device);
        let asym_id = Tensor::<B, 2, Int>::zeros([b, n], &device);
        let mol_type = Tensor::<B, 2, Int>::zeros([b, n], &device);
        let token_to_rep_atom = Tensor::<B, 2>::eye(n, &device).unsqueeze_dim::<3>(0);
        let frames_idx = Tensor::<B, 3, Int>::zeros([b, n, 3], &device);
        let pred_distogram_logits = Tensor::<B, 4>::random(
            [b, n, n, 32],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let out = m.forward(
            s_inputs,
            s,
            z,
            x_pred,
            token_pad_mask,
            asym_id,
            mol_type,
            token_to_rep_atom,
            frames_idx,
            pred_distogram_logits,
            1,
        );
        assert_eq!(out.pae_logits.dims(), [b, n, n, cfg.num_pae_bins]);
        assert_eq!(out.plddt.dims(), [b, n]);
        assert_eq!(out.complex_plddt.dims(), [b]);
        assert_eq!(out.ptm.dims(), [b]);
    }
}
