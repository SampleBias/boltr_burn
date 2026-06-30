//! Affinity head — `boltz.model.modules.affinity` (`AffinityModule`, `AffinityHeadsTransformer`).

use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, Linear, LinearConfig};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Int, Tensor};

use crate::burn_compat::{layer_norm_1d, linear_no_bias};
use crate::layers::PairformerNoSeqModule;
use crate::tensor_ops::{
    cdist_euclidean, dist_to_bin_indices, linspace, repeat_interleave_dim0,
    repeat_interleave_dim0_int,
};

use super::encoders::PairwiseConditioning;

pub const AFFINITY_MW_MODEL_COEF: f64 = 1.03525938;
pub const AFFINITY_MW_COEF: f64 = -0.59992683;
pub const AFFINITY_MW_BIAS: f64 = 2.83288489;

#[must_use]
pub fn apply_affinity_mw_correction<B: Backend>(
    affinity_pred_value: Tensor<B, 2>,
    affinity_mw: Tensor<B, 1>,
) -> Tensor<B, 2> {
    let mw_pow = affinity_mw.powf_scalar(0.3).unsqueeze_dim::<2>(1);
    affinity_pred_value.clone().mul_scalar(AFFINITY_MW_MODEL_COEF)
        + mw_pow.mul_scalar(AFFINITY_MW_COEF)
        + Tensor::<B, 2>::ones(affinity_pred_value.dims(), &affinity_pred_value.device())
            .mul_scalar(AFFINITY_MW_BIAS)
}

#[derive(Debug, Clone)]
pub struct AffinityModuleConfig {
    pub num_dist_bins: usize,
    pub max_dist: f64,
    pub pairformer_num_blocks: usize,
    pub pairformer_dropout: f64,
    pub pairformer_pairwise_head_width: usize,
    pub pairformer_pairwise_num_heads: usize,
    pub pairformer_post_layer_norm: bool,
    pub pairformer_activation_checkpointing: bool,
    pub head_token_s: usize,
}

impl Default for AffinityModuleConfig {
    fn default() -> Self {
        Self {
            num_dist_bins: 64,
            max_dist: 22.0,
            pairformer_num_blocks: 4,
            pairformer_dropout: 0.25,
            pairformer_pairwise_head_width: 32,
            pairformer_pairwise_num_heads: 4,
            pairformer_post_layer_norm: false,
            pairformer_activation_checkpointing: false,
            head_token_s: 384,
        }
    }
}

impl AffinityModuleConfig {
    #[must_use]
    pub fn from_affinity_model_args(v: Option<&serde_json::Value>, token_s: usize) -> Self {
        let mut cfg = Self {
            head_token_s: token_s,
            ..Default::default()
        };
        let Some(v) = v else {
            return cfg;
        };
        if let Some(n) = v.get("num_dist_bins").and_then(serde_json::Value::as_u64) {
            cfg.num_dist_bins = n as usize;
        }
        if let Some(x) = v.get("max_dist").and_then(|x| x.as_f64()) {
            cfg.max_dist = x;
        }
        if let Some(p) = v.get("pairformer_args").and_then(|x| x.as_object()) {
            if let Some(n) = p.get("num_blocks").and_then(serde_json::Value::as_u64) {
                cfg.pairformer_num_blocks = n as usize;
            }
            if let Some(d) = p.get("dropout").and_then(|x| x.as_f64()) {
                cfg.pairformer_dropout = d;
            }
            if let Some(w) = p
                .get("pairwise_head_width")
                .and_then(serde_json::Value::as_u64)
            {
                cfg.pairformer_pairwise_head_width = w as usize;
            }
            if let Some(h) = p
                .get("pairwise_num_heads")
                .and_then(serde_json::Value::as_u64)
            {
                cfg.pairformer_pairwise_num_heads = h as usize;
            }
            if let Some(b) = p
                .get("post_layer_norm")
                .and_then(serde_json::Value::as_bool)
            {
                cfg.pairformer_post_layer_norm = b;
            }
            if let Some(b) = p
                .get("activation_checkpointing")
                .and_then(serde_json::Value::as_bool)
            {
                cfg.pairformer_activation_checkpointing = b;
            }
        }
        if let Some(t) = v.get("transformer_args").and_then(|x| x.as_object()) {
            if let Some(ts) = t.get("token_s").and_then(serde_json::Value::as_u64) {
                cfg.head_token_s = ts as usize;
            }
        }
        cfg
    }
}

#[derive(Debug)]
pub struct AffinityOutput<B: Backend> {
    pub affinity_pred_value: Tensor<B, 2>,
    pub affinity_logits_binary: Tensor<B, 2>,
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

#[derive(Module, Debug)]
struct AffinityHeads<B: Backend> {
    out_lin1: Linear<B>,
    out_lin2: Linear<B>,
    pred_value_lin1: Linear<B>,
    pred_value_lin2: Linear<B>,
    pred_value_lin3: Linear<B>,
    pred_score_lin1: Linear<B>,
    pred_score_lin2: Linear<B>,
    pred_score_lin3: Linear<B>,
    logits_binary: Linear<B>,
}

impl<B: Backend> AffinityHeads<B> {
    fn new(device: &Device<B>, token_z: usize, head_s: usize) -> Self {
        Self {
            out_lin1: LinearConfig::new(token_z, token_z).init(device),
            out_lin2: LinearConfig::new(token_z, head_s).init(device),
            pred_value_lin1: LinearConfig::new(head_s, head_s).init(device),
            pred_value_lin2: LinearConfig::new(head_s, head_s).init(device),
            pred_value_lin3: LinearConfig::new(head_s, 1).init(device),
            pred_score_lin1: LinearConfig::new(head_s, head_s).init(device),
            pred_score_lin2: LinearConfig::new(head_s, head_s).init(device),
            pred_score_lin3: LinearConfig::new(head_s, 1).init(device),
            logits_binary: LinearConfig::new(1, 1).init(device),
        }
    }

    fn forward(
        &self,
        z: Tensor<B, 4>,
        token_pad_mask: Tensor<B, 2>,
        mol_type: Tensor<B, 2, Int>,
        affinity_token_mask: Tensor<B, 2>,
        multiplicity: usize,
    ) -> AffinityOutput<B> {
        let mut z = z;
        let pad_token_mask = repeat_interleave_dim0(token_pad_mask, multiplicity);
        let mol_b = repeat_interleave_dim0_int(mol_type, multiplicity);
        let rec_mask = mol_b.clone().equal_elem(0).float() * pad_token_mask.clone();
        let lig_mask = repeat_interleave_dim0(affinity_token_mask, multiplicity) * pad_token_mask;

        let mut cross_pair_mask = lig_mask.clone().unsqueeze_dim::<3>(2)
            * rec_mask.clone().unsqueeze_dim::<3>(1)
            + rec_mask.clone().unsqueeze_dim::<3>(2) * lig_mask.clone().unsqueeze_dim::<3>(1)
            + lig_mask.clone().unsqueeze_dim::<3>(2) * lig_mask.clone().unsqueeze_dim::<3>(1);

        let n = lig_mask.dims()[1];
        let device = lig_mask.device();
        let eye = Tensor::<B, 2>::eye(n, &device).unsqueeze_dim::<3>(0);
        cross_pair_mask = cross_pair_mask * (Tensor::<B, 3>::ones([1, n, n], &device) - eye);

        if multiplicity == 1 && z.dims().len() == 4 {
            let [b0, n1, n2, c] = z.dims();
            if b0 > 1 && b0 == n1 && n1 == n2 {
                z = z.slice([0..1, 0..n1, 0..n2, 0..c]);
            }
        }

        let [batch, _, _, channels] = z.dims();
        let numer: Tensor<B, 2> = (z.clone() * cross_pair_mask.clone().unsqueeze_dim::<4>(3))
            .reshape([batch, n * n, channels])
            .sum_dim(1)
            .squeeze_dim::<2>(1);
        let denom = cross_pair_mask
            .reshape([batch, n * n])
            .sum_dim(1)
            + 1e-7;
        let g = numer / denom.clone();

        let g = relu(self.out_lin2.forward(relu(self.out_lin1.forward(g))));

        let affinity_pred_value = self
            .pred_value_lin3
            .forward(relu(
                self.pred_value_lin2
                    .forward(relu(self.pred_value_lin1.forward(g.clone()))),
            ))
            .reshape([batch, 1]);
        let affinity_pred_score = self
            .pred_score_lin3
            .forward(relu(
                self.pred_score_lin2
                    .forward(relu(self.pred_score_lin1.forward(g))),
            ))
            .reshape([batch, 1]);
        let affinity_logits_binary = self
            .logits_binary
            .forward(affinity_pred_score)
            .reshape([batch, 1]);

        AffinityOutput {
            affinity_pred_value,
            affinity_logits_binary,
        }
    }
}

#[derive(Module, Debug)]
pub struct AffinityModule<B: Backend> {
    boundaries: Vec<f64>,
    dist_bin_pairwise_embed: Embedding<B>,
    s_to_z_prod_in1: Linear<B>,
    s_to_z_prod_in2: Linear<B>,
    z_norm: LayerNorm<B>,
    z_linear: Linear<B>,
    pairwise_conditioner: PairwiseConditioning<B>,
    pairformer_stack: PairformerNoSeqModule<B>,
    affinity_heads: AffinityHeads<B>,
}

impl<B: Backend> AffinityModule<B> {
    pub fn new(
        device: &Device<B>,
        token_s: usize,
        token_z: usize,
        cfg: &AffinityModuleConfig,
    ) -> Self {
        let num_dist = cfg.num_dist_bins;
        let boundaries = linspace(2.0, cfg.max_dist, num_dist.saturating_sub(1));

        Self {
            boundaries,
            dist_bin_pairwise_embed: EmbeddingConfig::new(num_dist, token_z).init(device),
            s_to_z_prod_in1: linear_no_bias(device, token_s, token_z),
            s_to_z_prod_in2: linear_no_bias(device, token_s, token_z),
            z_norm: layer_norm_1d(device, token_z),
            z_linear: linear_no_bias(device, token_z, token_z),
            pairwise_conditioner: PairwiseConditioning::new(device, token_z, token_z, 2, 2),
            pairformer_stack: PairformerNoSeqModule::new(
                device,
                token_z,
                cfg.pairformer_num_blocks,
                cfg.pairformer_dropout,
                cfg.pairformer_pairwise_head_width,
                cfg.pairformer_pairwise_num_heads,
            ),
            affinity_heads: AffinityHeads::new(device, token_z, cfg.head_token_s),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        s_inputs: Tensor<B, 3>,
        z: Tensor<B, 4>,
        x_pred: Tensor<B, 3>,
        token_pad_mask: Tensor<B, 2>,
        mol_type: Tensor<B, 2, Int>,
        affinity_token_mask: Tensor<B, 2>,
        token_to_rep_atom: Tensor<B, 3>,
        multiplicity: usize,
        use_kernels: bool,
    ) -> AffinityOutput<B> {
        let mut z = self.z_linear.forward(self.z_norm.forward(z));
        z = repeat_interleave_dim0(z, multiplicity);

        z = z
            + self.s_to_z_prod_in1.forward(s_inputs.clone()).unsqueeze_dim::<4>(2)
            + self
                .s_to_z_prod_in2
                .forward(s_inputs)
                .unsqueeze_dim::<4>(1);

        let token_to_rep_atom = repeat_interleave_dim0(token_to_rep_atom, multiplicity);

        let x_pred_repr = token_to_rep_atom.matmul(x_pred);
        let d = cdist_euclidean(x_pred_repr);
        let distogram = dist_to_bin_indices(d, &self.boundaries);
        let distogram = embed_dist_bins(&self.dist_bin_pairwise_embed, distogram);
        let z_pc = z.clone();
        z = z + self.pairwise_conditioner.forward(z_pc, distogram);

        let pad_token_mask = repeat_interleave_dim0(token_pad_mask.clone(), multiplicity);
        let mol_b = repeat_interleave_dim0_int(mol_type.clone(), multiplicity);
        let rec_mask = mol_b.clone().equal_elem(0).float() * pad_token_mask.clone();
        let lig_mask =
            repeat_interleave_dim0(affinity_token_mask.clone(), multiplicity) * pad_token_mask;

        let cross_pair_mask = lig_mask.clone().unsqueeze_dim::<3>(2)
            * rec_mask.clone().unsqueeze_dim::<3>(1)
            + rec_mask.clone().unsqueeze_dim::<3>(2) * lig_mask.clone().unsqueeze_dim::<3>(1)
            + lig_mask.clone().unsqueeze_dim::<3>(2) * lig_mask.clone().unsqueeze_dim::<3>(1);
        let pair_mask = cross_pair_mask;

        let z = self.pairformer_stack.forward(z, pair_mask, use_kernels);

        self.affinity_heads.forward(
            z,
            token_pad_mask,
            mol_type,
            affinity_token_mask,
            multiplicity,
        )
    }
}

pub type AffinityHead<B> = AffinityModule<B>;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::prelude::ElementConversion;

    type B = NdArray;

    #[test]
    fn mw_correction_matches_formula() {
        let device = Default::default();
        let pred = Tensor::<B, 2>::from_data([[0.5f32], [-1.0f32]], &device);
        let mw = Tensor::<B, 1>::from_data([400.0f32, 100.0f32], &device);
        let out = apply_affinity_mw_correction(pred, mw);
        let v0 = 0.5_f64;
        let mw0 = 400_f64.powf(0.3);
        let e0 = AFFINITY_MW_MODEL_COEF * v0 + AFFINITY_MW_COEF * mw0 + AFFINITY_MW_BIAS;
        let v1 = -1.0_f64;
        let mw1 = 100_f64.powf(0.3);
        let e1 = AFFINITY_MW_MODEL_COEF * v1 + AFFINITY_MW_COEF * mw1 + AFFINITY_MW_BIAS;
        let g0 = out.clone().slice([0..1, 0..1]).into_scalar().elem::<f64>();
        let g1 = out.slice([1..2, 0..1]).into_scalar().elem::<f64>();
        assert!((g0 - e0).abs() < 1e-5, "got {g0} expected {e0}");
        assert!((g1 - e1).abs() < 1e-5, "got {g1} expected {e1}");
    }

    #[test]
    fn affinity_module_forward_smoke() {
        let device = Default::default();
        let token_s = 32;
        let token_z = 16;
        let n = 8;
        let a = 8;
        let cfg = AffinityModuleConfig {
            num_dist_bins: 32,
            max_dist: 22.0,
            pairformer_num_blocks: 1,
            pairformer_dropout: 0.0,
            pairformer_pairwise_head_width: 8,
            pairformer_pairwise_num_heads: 2,
            pairformer_post_layer_norm: false,
            pairformer_activation_checkpointing: false,
            head_token_s: token_s,
        };

        let m = AffinityModule::<B>::new(&device, token_s, token_z, &cfg);

        let b = 1;
        let s_inputs = Tensor::<B, 3>::random(
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
        let mol_type = Tensor::cat(
            vec![
                Tensor::<B, 2, Int>::zeros([b, 6], &device),
                Tensor::<B, 2, Int>::full([b, 2], 2, &device),
            ],
            1,
        );
        let affinity_token_mask = Tensor::cat(
            vec![
                Tensor::<B, 2>::zeros([b, 6], &device),
                Tensor::<B, 2>::ones([b, 2], &device),
            ],
            1,
        );
        let token_to_rep_atom = Tensor::<B, 2>::eye(n, &device).unsqueeze_dim::<3>(0);

        let out = m.forward(
            s_inputs,
            z,
            x_pred,
            token_pad_mask,
            mol_type,
            affinity_token_mask,
            token_to_rep_atom,
            1,
            false,
        );
        assert_eq!(out.affinity_pred_value.dims(), [b, 1]);
        assert_eq!(out.affinity_logits_binary.dims(), [b, 1]);
        let v = out
            .affinity_pred_value
            .clone()
            .slice([0..1, 0..1])
            .into_scalar()
            .elem::<f32>();
        assert!(v.is_finite());
    }

    #[test]
    fn affinity_config_from_json() {
        let j = serde_json::json!({
            "pairformer_args": { "num_blocks": 2, "dropout": 0.1 },
            "transformer_args": { "token_s": 128 },
            "num_dist_bins": 48,
        });
        let c = AffinityModuleConfig::from_affinity_model_args(Some(&j), 384);
        assert_eq!(c.pairformer_num_blocks, 2);
        assert!((c.pairformer_dropout - 0.1).abs() < 1e-9);
        assert_eq!(c.head_token_s, 128);
        assert_eq!(c.num_dist_bins, 48);
    }
}
