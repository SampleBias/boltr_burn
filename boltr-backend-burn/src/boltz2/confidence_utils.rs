//! Helpers aligned with `boltz-reference/src/boltz/model/layers/confidence_utils.py`.

use std::collections::BTreeMap;

use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Int, Tensor};

use crate::tensor_ops::{repeat_interleave_dim0, repeat_interleave_dim0_int};

pub const CHAIN_TYPE_PROTEIN: i64 = 0;
pub const CHAIN_TYPE_NONPOLYMER: i64 = 3;

fn bin_centers<B: Backend>(num_bins: usize, bin_width: f64, device: &Device<B>) -> Tensor<B, 1> {
    Tensor::<B, 1, Int>::arange(0..num_bins as i64, device)
        .float()
        .mul_scalar(bin_width)
        .add_scalar(0.5 * bin_width)
}

/// `compute_aggregated_metric(logits, end)` — expected value over softmax bins.
pub fn compute_aggregated_metric<B: Backend>(
    logits: Tensor<B, 4>,
    end: f64,
) -> Tensor<B, 3> {
    let num_bins = logits.dims()[3];
    let bin_width = end / num_bins as f64;
    let bounds = bin_centers::<B>(num_bins, bin_width, &logits.device()).reshape([1, 1, 1, num_bins]);
    let probs = softmax(logits, 3);
    (probs * bounds).sum_dim(3).squeeze_dim::<3>(3)
}

/// Token-level variant: `[B, N, bins]` → `[B, N]`.
pub fn compute_aggregated_metric_token<B: Backend>(
    logits: Tensor<B, 3>,
    end: f64,
) -> Tensor<B, 2> {
    let num_bins = logits.dims()[2];
    let bin_width = end / num_bins as f64;
    let bounds = bin_centers::<B>(num_bins, bin_width, &logits.device()).reshape([1, 1, num_bins]);
    let probs = softmax(logits, 2);
    (probs * bounds).sum_dim(2).squeeze_dim::<2>(2)
}

fn tm_function<B: Backend>(d: Tensor<B, 2>, n_res: Tensor<B, 2>) -> Tensor<B, 2> {
    let d0 = n_res
        .clamp_min(19.0)
        .sub_scalar(15.0)
        .powf_scalar(1.0 / 3.0)
        .mul_scalar(1.24)
        .sub_scalar(1.8);
    let ratio = d / d0;
    Tensor::<B, 2>::ones(ratio.dims(), &ratio.device())
        / (Tensor::<B, 2>::ones(ratio.dims(), &ratio.device()) + ratio.powf_scalar(2.0))
}

pub fn compute_frame_pred_stub<B: Backend>(
    frames_idx_true: Tensor<B, 3, Int>,
    token_pad_mask: Tensor<B, 2>,
    multiplicity: usize,
) -> (Tensor<B, 4, Int>, Tensor<B, 3>) {
    let b_total = frames_idx_true.dims()[0];
    let n_atom = frames_idx_true.dims()[1];
    let b0 = b_total / multiplicity;
    let frames_idx_pred = repeat_interleave_dim0_int(frames_idx_true, multiplicity)
        .reshape([b0, multiplicity, n_atom, 3]);
    let n_tok = token_pad_mask.dims()[1];
    let mask_collinear = token_pad_mask
        .reshape([b0, 1, n_tok])
        .repeat(&[1, multiplicity, 1]);
    (frames_idx_pred, mask_collinear)
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn compute_ptms<B: Backend>(
    pae_logits: Tensor<B, 4>,
    _x_preds: Tensor<B, 3>,
    frames_idx: Tensor<B, 3, Int>,
    asym_id: Tensor<B, 2, Int>,
    mol_type: Tensor<B, 2, Int>,
    token_pad_mask: Tensor<B, 2>,
    multiplicity: usize,
) -> (
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    BTreeMap<i64, BTreeMap<i64, Tensor<B, 1>>>,
) {
    let (_frames_idx_pred, mask_collinear_pred) =
        compute_frame_pred_stub(frames_idx, token_pad_mask.clone(), multiplicity);

    let mask_pad = repeat_interleave_dim0(token_pad_mask.clone(), multiplicity);
    let n = mask_collinear_pred.dims()[2];
    let maski_rows = mask_collinear_pred.dims()[0] * multiplicity;
    let maski = mask_collinear_pred.reshape([maski_rows, n]);

    let pad_pair = mask_pad.clone().unsqueeze_dim::<3>(2) * mask_pad.clone().unsqueeze_dim::<3>(1);
    let pair_mask_ptm = maski.clone().unsqueeze_dim::<3>(2) * pad_pair.clone();

    let asym_r = repeat_interleave_dim0_int(asym_id.clone(), multiplicity);
    let ne = asym_r
        .clone()
        .unsqueeze_dim::<3>(2)
        .not_equal(asym_r.clone().unsqueeze_dim::<3>(1))
        .float();
    let pair_mask_iptm = maski.clone().unsqueeze_dim::<3>(2) * ne.clone() * pad_pair.clone();

    let num_bins = pae_logits.dims()[3];
    let bin_width = 32.0 / num_bins as f64;
    let device = pae_logits.device();
    let pae_value = bin_centers::<B>(num_bins, bin_width, &device).reshape([1, num_bins]);
    let n_res = mask_pad.sum_dim(1);
    let tm_w = tm_function(pae_value, n_res);
    let tm_value = tm_w.unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(2);
    let probs = softmax(pae_logits, 3);
    let tm_expected_value: Tensor<B, 3> = (probs * tm_value).sum_dim(3).squeeze_dim::<3>(3);

    let eps = 1e-5;
    let ptm: Tensor<B, 2> = (tm_expected_value.clone() * pair_mask_ptm.clone())
        .sum_dim(2)
        .squeeze_dim::<2>(2)
        / (pair_mask_ptm.sum_dim(2).squeeze_dim::<2>(2) + eps);
    let ptm = ptm.max_dim(1).squeeze_dim::<1>(1);

    let iptm: Tensor<B, 2> = (tm_expected_value.clone() * pair_mask_iptm.clone())
        .sum_dim(2)
        .squeeze_dim::<2>(2)
        / (pair_mask_iptm.sum_dim(2).squeeze_dim::<2>(2) + eps);
    let iptm = iptm.max_dim(1).squeeze_dim::<1>(1);

    let is_ligand = repeat_interleave_dim0_int(mol_type.clone(), multiplicity)
        .equal_elem(CHAIN_TYPE_NONPOLYMER)
        .float();
    let is_protein = repeat_interleave_dim0_int(mol_type, multiplicity)
        .equal_elem(CHAIN_TYPE_PROTEIN)
        .float();

    let ligand_iptm_mask = maski.clone().unsqueeze_dim::<3>(2)
        * ne.clone()
        * pad_pair.clone()
        * ((is_ligand.clone().unsqueeze_dim::<3>(2) * is_protein.clone().unsqueeze_dim::<3>(1))
            + (is_protein.clone().unsqueeze_dim::<3>(2) * is_ligand.clone().unsqueeze_dim::<3>(1)));
    let protein_iptm_mask = maski.clone().unsqueeze_dim::<3>(2)
        * ne
        * pad_pair.clone()
        * (is_protein.clone().unsqueeze_dim::<3>(2) * is_protein.clone().unsqueeze_dim::<3>(1));

    let ligand_iptm = ((tm_expected_value.clone() * ligand_iptm_mask.clone())
        .sum_dim(2)
        .squeeze_dim::<2>(2)
        / (ligand_iptm_mask.sum_dim(2).squeeze_dim::<2>(2) + eps))
        .max_dim(1)
        .squeeze_dim::<1>(1);
    let protein_iptm = ((tm_expected_value.clone() * protein_iptm_mask.clone())
        .sum_dim(2)
        .squeeze_dim::<2>(2)
        / (protein_iptm_mask.sum_dim(2).squeeze_dim::<2>(2) + eps))
        .max_dim(1)
        .squeeze_dim::<1>(1);

    let mut pair_chains: BTreeMap<i64, BTreeMap<i64, Tensor<B, 1>>> = BTreeMap::new();
    let flat = asym_r.clone().reshape([asym_r.dims()[0] * asym_r.dims()[1]]);
    let mut uniq_ids: Vec<i64> = flat.into_data().to_vec::<i64>().unwrap_or_default();
    uniq_ids.sort_unstable();
    uniq_ids.dedup();

    for &idx1 in &uniq_ids {
        let mut inner = BTreeMap::new();
        for &idx2 in &uniq_ids {
            let mask_pair_chain = maski.clone().unsqueeze_dim::<3>(2)
                * asym_r
                    .clone()
                    .unsqueeze_dim::<3>(2)
                    .equal_elem(idx1)
                    .float()
                * asym_r
                    .clone()
                    .unsqueeze_dim::<3>(1)
                    .equal_elem(idx2)
                    .float()
                * pad_pair.clone();
            let v = ((tm_expected_value.clone() * mask_pair_chain.clone())
                .sum_dim(2)
                .squeeze_dim::<2>(2)
                / (mask_pair_chain.sum_dim(2).squeeze_dim::<2>(2) + eps))
                .max_dim(1)
                .squeeze_dim::<1>(1);
            inner.insert(idx2, v);
        }
        pair_chains.insert(idx1, inner);
    }

    (ptm, iptm, ligand_iptm, protein_iptm, pair_chains)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn aggregated_metric_shape() {
        let device = Default::default();
        let logits = Tensor::<B, 4>::random(
            [2, 6, 6, 8],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let out = compute_aggregated_metric(logits, 32.0);
        assert_eq!(out.dims(), [2, 6, 6]);
    }

    #[test]
    fn compute_ptms_smoke() {
        let device = Default::default();
        let b = 1;
        let n = 6;
        let m = 1;
        let pae_logits = Tensor::<B, 4>::random(
            [b * m, n, n, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let x_pred = Tensor::<B, 3>::random(
            [b * m, n, 3],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let frames_idx = Tensor::<B, 3, Int>::zeros([b * m, n, 3], &device);
        let asym_id = Tensor::<B, 2, Int>::zeros([b, n], &device);
        let mol_type = Tensor::<B, 2, Int>::zeros([b, n], &device);
        let token_pad_mask = Tensor::<B, 2>::ones([b, n], &device);
        let (ptm, iptm, lig_iptm, prot_iptm, pair_chains) = compute_ptms(
            pae_logits,
            x_pred,
            frames_idx,
            asym_id,
            mol_type,
            token_pad_mask,
            m,
        );
        assert_eq!(ptm.dims(), [b * m]);
        assert_eq!(iptm.dims(), [b * m]);
        assert_eq!(lig_iptm.dims(), [b * m]);
        assert_eq!(prot_iptm.dims(), [b * m]);
        assert!(!pair_chains.is_empty());
    }
}
