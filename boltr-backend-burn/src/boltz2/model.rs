//! Boltz2 high-level model on Burn `Module` types.
//!
//! Python layout reference: `boltz-reference/src/boltz/model/models/boltz2.py`
//!
//! Parameter paths match Lightning `state_dict` keys (same as `boltr-backend-tch` VarStore).

use std::path::Path;

use anyhow::{Context, Result};
use boltr_backend_core::{
    expected_keys_missing_in_safetensors, safetensor_names_not_in_expected, Boltz2Hparams,
    Boltz2ModelDims,
};
use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Int, Tensor};
use burn_store::{BurnToPyTorchAdapter, ModuleSnapshot};

use super::affinity::{AffinityModule, AffinityModuleConfig};
use super::confidence::{ConfidenceModule, ConfidenceModuleConfig};
use super::diffusion::{AtomDiffusion, AtomDiffusionConfig, DiffusionSampleOutput};
use super::diffusion_conditioning::{DiffusionConditioning, DiffusionConditioningOutput};
use super::distogram::{BFactorModule, DistogramModule};
use super::encoders::{AtomEncoderBatchFeats, AtomEncoderFlags};
use super::input_embedder::InputEmbedder;
use super::msa_module::MsaFeatures;
use super::relative_position::{RelPosFeatures, RelativePositionEncoder};
use super::steering::SteeringParams;
use super::template_module::TemplateFeatures;
use super::trunk::TrunkV2;

/// `len(const.bond_types) + 1` in Boltz (`boltz-reference/.../const.py`).
pub const BOND_TYPE_EMBEDDING_NUM: usize = 7;

/// Configuration for sizing the Burn inference graph.
#[derive(Debug, Clone)]
pub struct Boltz2BurnModelConfig {
    pub dims: Boltz2ModelDims,
    pub atom_s: i64,
    pub atom_z: i64,
    pub num_bins: i64,
}

impl Boltz2BurnModelConfig {
    #[must_use]
    pub fn from_hparams(h: &Boltz2Hparams) -> Self {
        Self {
            dims: Boltz2ModelDims::from_hparams(h),
            atom_s: h.atom_s.unwrap_or(128),
            atom_z: h.atom_z.unwrap_or(16),
            num_bins: h.num_bins.unwrap_or(64),
        }
    }

    #[must_use]
    pub fn with_defaults(token_s: i64, token_z: i64, num_blocks: Option<i64>) -> Self {
        let h = Boltz2Hparams {
            token_s: Some(token_s),
            token_z: Some(token_z),
            num_blocks,
            ..Default::default()
        };
        Self::from_hparams(&h)
    }
}

/// Boltz2 default score model / diffusion hyper-parameters (matching Python defaults).
#[derive(Debug, Clone)]
pub struct Boltz2DiffusionArgs {
    pub atom_s: usize,
    pub atom_z: usize,
    pub atoms_per_window_queries: usize,
    pub atoms_per_window_keys: usize,
    pub atom_encoder_depth: usize,
    pub atom_encoder_heads: usize,
    pub token_transformer_depth: usize,
    pub token_transformer_heads: usize,
    pub atom_decoder_depth: usize,
    pub atom_decoder_heads: usize,
    pub atom_feature_dim: usize,
    pub atom_encoder_flags: AtomEncoderFlags,
    pub conditioning_transition_layers: usize,
    pub dim_fourier: usize,
    pub num_bins: usize,
    pub predict_bfactor: bool,
    pub use_templates: Option<(usize, usize)>,
}

impl Default for Boltz2DiffusionArgs {
    fn default() -> Self {
        let atom_encoder_flags = AtomEncoderFlags::default();
        let atom_feature_dim = atom_encoder_flags.expected_atom_feature_dim();
        Self {
            atom_s: 128,
            atom_z: 16,
            atoms_per_window_queries: 32,
            atoms_per_window_keys: 128,
            atom_encoder_depth: 3,
            atom_encoder_heads: 4,
            token_transformer_depth: 24,
            token_transformer_heads: 8,
            atom_decoder_depth: 3,
            atom_decoder_heads: 4,
            atom_feature_dim,
            atom_encoder_flags,
            conditioning_transition_layers: 2,
            dim_fourier: 256,
            num_bins: 64,
            predict_bfactor: false,
            use_templates: None,
        }
    }
}

impl Boltz2DiffusionArgs {
    /// Minimal diffusion args for unit/integration smoke tests (small token/atom counts).
    #[must_use]
    pub fn tiny_for_tests(num_bins: usize) -> Self {
        let atom_encoder_flags = AtomEncoderFlags {
            num_elements: 4,
            ..AtomEncoderFlags::default()
        };
        let atom_feature_dim = atom_encoder_flags.expected_atom_feature_dim();
        Self {
            atom_s: 16,
            atom_z: 8,
            atoms_per_window_queries: 4,
            atoms_per_window_keys: 8,
            atom_encoder_depth: 1,
            atom_encoder_heads: 2,
            token_transformer_depth: 1,
            token_transformer_heads: 2,
            atom_decoder_depth: 1,
            atom_decoder_heads: 2,
            atom_feature_dim,
            atom_encoder_flags,
            num_bins,
            ..Default::default()
        }
    }

    /// Merge [`Boltz2Hparams`] into defaults so layer sizes match exported checkpoints.
    #[must_use]
    pub fn from_boltz2_hparams(h: &Boltz2Hparams) -> Self {
        let mut d = Self::default();
        if let Some(atom_s) = h.atom_s {
            d.atom_s = atom_s as usize;
        }
        if let Some(atom_z) = h.atom_z {
            d.atom_z = atom_z as usize;
        }
        if let Some(n) = h.num_bins {
            d.num_bins = n as usize;
        }
        if let Some(pb) = h.other.get("predict_bfactor").and_then(|v| v.as_bool()) {
            d.predict_bfactor = pb;
        }
        if let Some(b) = h.other.get("use_no_atom_char").and_then(|v| v.as_bool()) {
            d.atom_encoder_flags.use_no_atom_char = b;
        }
        if let Some(b) = h
            .other
            .get("use_atom_backbone_feat")
            .and_then(|v| v.as_bool())
        {
            d.atom_encoder_flags.use_atom_backbone_feat = b;
        }
        if let Some(b) = h
            .other
            .get("use_residue_feats_atoms")
            .and_then(|v| v.as_bool())
        {
            d.atom_encoder_flags.use_residue_feats_atoms = b;
        }
        if let Some(v) = &h.score_model_args {
            if let Some(obj) = v.as_object() {
                if let Some(x) = obj.get("atom_encoder_depth").and_then(|x| x.as_i64()) {
                    d.atom_encoder_depth = x as usize;
                }
                if let Some(x) = obj.get("atom_encoder_heads").and_then(|x| x.as_i64()) {
                    d.atom_encoder_heads = x as usize;
                }
                if let Some(x) = obj.get("atom_decoder_depth").and_then(|x| x.as_i64()) {
                    d.atom_decoder_depth = x as usize;
                }
                if let Some(x) = obj.get("atom_decoder_heads").and_then(|x| x.as_i64()) {
                    d.atom_decoder_heads = x as usize;
                }
                if let Some(x) = obj.get("token_transformer_depth").and_then(|x| x.as_i64()) {
                    d.token_transformer_depth = x as usize;
                }
                if let Some(x) = obj.get("token_transformer_heads").and_then(|x| x.as_i64()) {
                    d.token_transformer_heads = x as usize;
                }
                if let Some(x) = obj
                    .get("conditioning_transition_layers")
                    .and_then(|x| x.as_i64())
                {
                    d.conditioning_transition_layers = x as usize;
                }
                if let Some(x) = obj.get("dim_fourier").and_then(|x| x.as_i64()) {
                    d.dim_fourier = x as usize;
                }
            }
        }
        d.atoms_per_window_queries = h.resolved_atoms_per_window_queries() as usize;
        d.atoms_per_window_keys = h.resolved_atoms_per_window_keys() as usize;
        d.atom_feature_dim = d.atom_encoder_flags.expected_atom_feature_dim();
        d
    }
}

/// Boltz2 inference model: trunk + pairformer + diffusion conditioning + structure module +
/// distogram/bfactor heads.
///
/// `contact_conditioning` is not wired (optional stub skipped until ported).
#[derive(Module, Debug)]
pub struct Boltz2BurnModel<B: Backend> {
    pub trunk: TrunkV2<B>,
    pub rel_pos: RelativePositionEncoder<B>,
    pub token_bonds: Linear<B>,
    pub token_bonds_type: Option<Embedding<B>>,
    pub input_embedder: InputEmbedder<B>,
    pub diffusion_conditioning: DiffusionConditioning<B>,
    pub structure_module: AtomDiffusion<B>,
    pub distogram_module: DistogramModule<B>,
    pub bfactor_module: Option<BFactorModule<B>>,
    pub confidence_module: Option<ConfidenceModule<B>>,
    pub affinity_module: Option<AffinityModule<B>>,
    #[module(skip)]
    affinity_mw_correction: bool,
    #[module(skip)]
    token_s: usize,
    #[module(skip)]
    token_z: usize,
    #[module(skip)]
    atom_s: usize,
    #[module(skip)]
    atom_z: usize,
    #[module(skip)]
    num_bins: usize,
}

fn remap_lightning_param_path(burn_path: &str) -> String {
    if let Some(rest) = burn_path.strip_prefix("trunk.") {
        let top = rest.split('.').next().unwrap_or("");
        return match top {
            "s_init" | "z_init_1" | "z_init_2" | "s_norm" | "z_norm" | "s_recycle" | "z_recycle" => {
                rest.to_string()
            }
            "pairformer" => burn_path.replacen("trunk.pairformer", "pairformer_module", 1),
            "msa" => burn_path.replacen("trunk.msa", "msa_module", 1),
            "template" => burn_path.replacen("trunk.template", "template_module", 1),
            _ => burn_path.to_string(),
        };
    }
    burn_path.to_string()
}

fn pytorch_param_name(path: &str, container_type: Option<&str>) -> String {
    let remapped = remap_lightning_param_path(path);
    if let Some(ct) = container_type {
        if ct.contains("LayerNorm") {
            if remapped.ends_with(".gamma") {
                return remapped.replace(".gamma", ".weight");
            }
            if remapped.ends_with(".beta") {
                return remapped.replace(".beta", ".bias");
            }
        }
    }
    remapped
}

impl<B: Backend> Boltz2BurnModel<B> {
    /// Build with default diffusion args derived from `config`.
    pub fn new(device: &Device<B>, config: &Boltz2BurnModelConfig) -> Self {
        let diff_args = Boltz2DiffusionArgs {
            atom_s: config.atom_s as usize,
            atom_z: config.atom_z as usize,
            num_bins: config.num_bins as usize,
            ..Default::default()
        };
        Self::with_all_options(
            device,
            config,
            diff_args,
            AtomDiffusionConfig::default(),
            None,
            None,
            false,
        )
    }

    /// Full constructor with all diffusion / head arguments.
    #[allow(clippy::too_many_arguments)]
    pub fn with_all_options(
        device: &Device<B>,
        config: &Boltz2BurnModelConfig,
        diff_args: Boltz2DiffusionArgs,
        diff_config: AtomDiffusionConfig,
        confidence: Option<ConfidenceModuleConfig>,
        affinity: Option<AffinityModuleConfig>,
        affinity_mw_correction: bool,
    ) -> Self {
        let token_s = config.dims.token_s as usize;
        let token_z = config.dims.token_z as usize;
        let num_blocks = config.dims.num_pairformer_blocks as usize;
        let bond_type_feature = config.dims.bond_type_feature;

        let trunk = TrunkV2::new(
            device,
            Some(token_s),
            Some(token_z),
            Some(num_blocks),
            diff_args.use_templates,
        );
        let rel_pos = RelativePositionEncoder::new(device, token_z, None, None, false, false);
        let token_bonds = LinearConfig::new(1, token_z)
            .with_bias(false)
            .init(device);
        let token_bonds_type = bond_type_feature.then(|| {
            EmbeddingConfig::new(BOND_TYPE_EMBEDDING_NUM, token_z).init(device)
        });
        let input_embedder = InputEmbedder::new_tail_only(device, token_s);
        let diffusion_conditioning = DiffusionConditioning::new(
            device,
            token_s,
            token_z,
            diff_args.atom_s,
            diff_args.atom_z,
            diff_args.atoms_per_window_queries,
            diff_args.atoms_per_window_keys,
            diff_args.atom_encoder_depth,
            diff_args.atom_encoder_heads,
            diff_args.token_transformer_depth,
            diff_args.token_transformer_heads,
            diff_args.atom_decoder_depth,
            diff_args.atom_decoder_heads,
            diff_args.atom_feature_dim,
            diff_args.conditioning_transition_layers,
            diff_args.atom_encoder_flags.clone(),
        );
        let structure_module = AtomDiffusion::new(
            device,
            token_s,
            diff_args.atom_s,
            diff_args.atoms_per_window_queries,
            diff_args.atoms_per_window_keys,
            diff_args.dim_fourier,
            diff_args.atom_encoder_depth,
            diff_args.atom_encoder_heads,
            diff_args.token_transformer_depth,
            diff_args.token_transformer_heads,
            diff_args.atom_decoder_depth,
            diff_args.atom_decoder_heads,
            diff_args.conditioning_transition_layers,
            diff_config,
        );
        let distogram_module =
            DistogramModule::new(device, token_z, diff_args.num_bins, None);
        let bfactor_module = diff_args
            .predict_bfactor
            .then(|| BFactorModule::new(device, token_s, diff_args.num_bins));
        let confidence_module = confidence.map(|mut cfg| {
            cfg.pairformer_num_blocks = num_blocks;
            ConfidenceModule::new(device, token_s, token_z, &cfg)
        });
        let affinity_module = affinity.map(|cfg| {
            AffinityModule::new(device, token_s, token_z, &cfg)
        });

        Self {
            trunk,
            rel_pos,
            token_bonds,
            token_bonds_type,
            input_embedder,
            diffusion_conditioning,
            structure_module,
            distogram_module,
            bfactor_module,
            confidence_module,
            affinity_module,
            affinity_mw_correction,
            token_s,
            token_z,
            atom_s: diff_args.atom_s,
            atom_z: diff_args.atom_z,
            num_bins: diff_args.num_bins,
        }
    }

    pub fn token_s(&self) -> usize {
        self.token_s
    }

    pub fn token_z(&self) -> usize {
        self.token_z
    }

    pub fn atom_s(&self) -> usize {
        self.atom_s
    }

    pub fn atom_z(&self) -> usize {
        self.atom_z
    }

    pub fn num_bins(&self) -> usize {
        self.num_bins
    }

    pub fn bond_type_feature(&self) -> bool {
        self.token_bonds_type.is_some()
    }

    pub fn affinity_mw_correction(&self) -> bool {
        self.affinity_mw_correction
    }

    /// Collect Burn parameter keys remapped to Lightning `state_dict` paths.
    pub fn parameter_names(&self) -> Vec<String> {
        let snapshots = self.collect(
            None,
            Some(Box::new(BurnToPyTorchAdapter)),
            true,
        );
        let mut names: Vec<String> = snapshots
            .iter()
            .map(|s| pytorch_param_name(&s.full_path(), s.module_type().as_deref()))
            .collect();
        names.sort();
        names.dedup();
        names
    }

    /// Top-level `state_dict` segments present in [`Self::parameter_names`].
    pub fn parameter_top_level_segments(&self) -> Vec<String> {
        let mut segs: Vec<String> = self
            .parameter_names()
            .iter()
            .filter_map(|k| k.split('.').next().map(str::to_string))
            .collect();
        segs.sort();
        segs.dedup();
        segs
    }

    /// Keys in the model graph absent from a safetensors file.
    pub fn keys_missing_in_safetensors(&self, path: &Path) -> Result<Vec<String>> {
        let names = self.parameter_names();
        expected_keys_missing_in_safetensors(path, &names)
    }

    /// Safetensors keys not mapped into this model (naming mismatch or unported modules).
    pub fn safetensors_keys_unused(&self, path: &Path) -> Result<Vec<String>> {
        let names = self.parameter_names();
        safetensor_names_not_in_expected(path, &names)
    }

    /// Load hparams JSON and construct a sized model.
    pub fn from_hparams_json(
        device: &Device<B>,
        path: &Path,
    ) -> Result<(Self, Boltz2BurnModelConfig)> {
        let raw = std::fs::read(path).with_context(|| format!("read hparams {}", path.display()))?;
        let h = Boltz2Hparams::from_json_slice(&raw)?;
        let config = Boltz2BurnModelConfig::from_hparams(&h);
        let confidence = if h.confidence_prediction == Some(false) {
            None
        } else {
            let mut c = ConfidenceModuleConfig::default();
            if let Some(nb) = h.resolved_num_pairformer_blocks() {
                c.pairformer_num_blocks = nb as usize;
            }
            Some(c)
        };
        let affinity = if h.affinity_prediction == Some(true) {
            Some(AffinityModuleConfig::from_affinity_model_args(
                h.affinity_model_args.as_ref(),
                h.resolved_token_s() as usize,
            ))
        } else {
            None
        };
        let affinity_mw_correction = h.affinity_mw_correction.unwrap_or(false);
        let diff_args = Boltz2DiffusionArgs::from_boltz2_hparams(&h);
        let diff_config = AtomDiffusionConfig::from_boltz2_hparams(&h);
        Ok((
            Self::with_all_options(
                device,
                &config,
                diff_args,
                diff_config,
                confidence,
                affinity,
                affinity_mw_correction,
            ),
            config,
        ))
    }

    pub fn forward_rel_pos(&self, rel: &RelPosFeatures<'_, B>) -> Tensor<B, 4> {
        self.rel_pos.forward(rel)
    }

    /// `z += token_bonds(feats["token_bonds"])` (+ optional type embedding).
    pub fn forward_token_bonds_bias(
        &self,
        batch: usize,
        num_tokens: usize,
        token_bonds: Option<Tensor<B, 4>>,
        type_bonds: Option<Tensor<B, 3, Int>>,
    ) -> Result<Tensor<B, 4>> {
        let device = self.token_bonds.weight.val().device();
        let tb_in = match token_bonds {
            Some(t) => t,
            None => Tensor::<B, 4>::zeros([batch, num_tokens, num_tokens, 1], &device),
        };
        let mut z = self.token_bonds.forward(tb_in);
        if let Some(emb) = &self.token_bonds_type {
            let idx = type_bonds.ok_or_else(|| {
                anyhow::anyhow!("type_bonds is required when bond_type_feature is enabled")
            })?;
            let [b, n, n2] = idx.dims();
            let flat = idx.reshape([b * n, n2]);
            let emb_out = emb.forward(flat).reshape([b, n, n2, self.token_z]);
            z = z + emb_out;
        }
        Ok(z)
    }

    /// Trunk forward with Python-aligned `z_init += rel_pos + token_bonds` before recycling.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_trunk_with_z_init_terms(
        &self,
        s_inputs: Tensor<B, 3>,
        rel: &RelPosFeatures<'_, B>,
        token_bonds: Option<Tensor<B, 4>>,
        type_bonds: Option<Tensor<B, 3, Int>>,
        token_pad_mask: Tensor<B, 2>,
        recycling_steps: Option<usize>,
        msa_feats: Option<&MsaFeatures<'_, B>>,
        template_feats: Option<&TemplateFeatures<'_, B>>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let [batch, num_tokens, _] = s_inputs.dims();
        let (s_init, z_pair) = self.trunk.initialize(s_inputs);
        let z_rel = self.rel_pos.forward(rel);
        let z_bonds = self
            .forward_token_bonds_bias(batch, num_tokens, token_bonds, type_bonds)
            .expect("token bonds bias");
        let z_init = z_pair + z_rel + z_bonds;
        self.trunk.forward_from_init(
            s_init,
            z_init,
            token_pad_mask,
            recycling_steps,
            msa_feats,
            template_feats,
        )
    }

    /// Run `DiffusionConditioning` on trunk outputs.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_diffusion_conditioning(
        &self,
        s_trunk: Tensor<B, 3>,
        z_trunk: Tensor<B, 4>,
        relative_position_encoding: Tensor<B, 4>,
        ref_pos: Tensor<B, 3>,
        ref_charge: Tensor<B, 2>,
        ref_element: Tensor<B, 3>,
        atom_pad_mask: Tensor<B, 2>,
        ref_space_uid: Tensor<B, 2, Int>,
        atom_to_token: Tensor<B, 3>,
        atom_encoder_batch: Option<&AtomEncoderBatchFeats<'_, B>>,
    ) -> DiffusionConditioningOutput<B> {
        self.diffusion_conditioning.forward(
            s_trunk,
            z_trunk,
            relative_position_encoding,
            ref_pos,
            ref_charge,
            ref_element,
            atom_pad_mask,
            ref_space_uid,
            atom_to_token,
            atom_encoder_batch,
        )
    }

    /// Run the distogram head on pair representation `z`.
    pub fn forward_distogram(&self, z: Tensor<B, 4>) -> Tensor<B, 5> {
        self.distogram_module.forward(z)
    }

    /// Run the B-factor head on single representation `s` (returns `None` if disabled).
    pub fn forward_bfactor(&self, s: Tensor<B, 3>) -> Option<Tensor<B, 3>> {
        self.bfactor_module.as_ref().map(|m| m.forward(s))
    }

    /// Run reverse-diffusion sampling (fast path when steering is `None` or [`SteeringParams::fast_path`]).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_diffusion_sample(
        &self,
        s_inputs: Tensor<B, 3>,
        s_trunk: Tensor<B, 3>,
        cond: &DiffusionConditioningOutput<B>,
        token_pad_mask: Tensor<B, 2>,
        atom_pad_mask: Tensor<B, 2>,
        atom_to_token: Tensor<B, 3>,
        num_sampling_steps: Option<usize>,
        multiplicity: usize,
        steering: Option<SteeringParams>,
    ) -> DiffusionSampleOutput<B> {
        match steering {
            None => self.structure_module.sample(
                s_inputs,
                s_trunk,
                cond,
                token_pad_mask,
                atom_pad_mask,
                atom_to_token,
                num_sampling_steps,
                multiplicity,
            ),
            Some(s) if !s.uses_extended_sampler() => self.structure_module.sample(
                s_inputs,
                s_trunk,
                cond,
                token_pad_mask,
                atom_pad_mask,
                atom_to_token,
                num_sampling_steps,
                multiplicity,
            ),
            Some(s) => self.structure_module.sample_with_steering(
                s_inputs,
                s_trunk,
                cond,
                token_pad_mask,
                atom_pad_mask,
                atom_to_token,
                num_sampling_steps,
                multiplicity,
                &s,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use boltr_backend_core::Boltz2ModelDims;
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Distribution;

    type B = NdArray;

    fn small_model(device: &Device<B>) -> Boltz2BurnModel<B> {
        let config = Boltz2BurnModelConfig {
            dims: Boltz2ModelDims {
                token_s: 64,
                token_z: 32,
                num_pairformer_blocks: 1,
                bond_type_feature: false,
            },
            atom_s: 16,
            atom_z: 8,
            num_bins: 8,
        };
        let diff_args = Boltz2DiffusionArgs::tiny_for_tests(8);
        Boltz2BurnModel::with_all_options(
            device,
            &config,
            diff_args,
            AtomDiffusionConfig::default(),
            None,
            None,
            false,
        )
    }

    #[test]
    fn parameter_names_include_major_submodules() {
        let device = Default::default();
        let model = small_model(&device);
        let names = model.parameter_names();
        assert!(names.iter().any(|k| k.starts_with("s_init")));
        assert!(names.iter().any(|k| k.starts_with("pairformer_module")));
        assert!(names.iter().any(|k| k.starts_with("diffusion_conditioning")));
        assert!(names.iter().any(|k| k.starts_with("structure_module.score_model")));
        assert!(names.iter().any(|k| k.starts_with("distogram_module")));
        assert!(names.iter().any(|k| k.starts_with("token_bonds")));
    }

    #[test]
    fn parses_hparams_fixture() {
        let device = Default::default();
        let p = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../boltr-backend-core/tests/fixtures/hparams/minimal.json");
        let (_model, cfg) = Boltz2BurnModel::<B>::from_hparams_json(&device, &p).unwrap();
        assert_eq!(cfg.dims.token_s, 384);
        assert_eq!(cfg.dims.token_z, 128);
    }

    #[test]
    fn predict_step_random_smoke() {
        let device = Default::default();
        let model = small_model(&device);
        let b = 1_usize;
        let n = 4_usize;
        let n_atoms = 4_usize;
        let token_s = model.token_s();
        let token_z = model.token_z();
        let num_bins = model.num_bins();

        let s_inputs = Tensor::<B, 3>::random(
            [b, n, token_s],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let asym_id = Tensor::<B, 2, Int>::zeros([b, n], &device);
        let residue_index =
            Tensor::<B, 1, Int>::arange(0..n as i64, &device).reshape([1, n]).repeat(&[b, 1]);
        let entity_id = Tensor::<B, 2, Int>::zeros([b, n], &device);
        let token_index = residue_index.clone();
        let sym_id = Tensor::<B, 2, Int>::zeros([b, n], &device);
        let cyclic_period = Tensor::<B, 2, Int>::zeros([b, n], &device);
        let rel = RelPosFeatures {
            asym_id: &asym_id,
            residue_index: &residue_index,
            entity_id: &entity_id,
            token_index: &token_index,
            sym_id: &sym_id,
            cyclic_period: &cyclic_period,
        };
        let pad = Tensor::<B, 2>::ones([b, n], &device);

        let (s_trunk, z_trunk) = model.forward_trunk_with_z_init_terms(
            s_inputs.clone(),
            &rel,
            None,
            None,
            pad.clone(),
            Some(0),
            None,
            None,
        );
        assert_eq!(s_trunk.dims(), [b, n, token_s]);
        assert_eq!(z_trunk.dims(), [b, n, n, token_z]);

        let rel_enc = model.forward_rel_pos(&rel);
        let ref_pos = Tensor::<B, 3>::random(
            [b, n_atoms, 3],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let ref_charge = Tensor::<B, 2>::random(
            [b, n_atoms],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let ref_element = Tensor::<B, 3>::random(
            [b, n_atoms, 4],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let atom_pad_mask = Tensor::<B, 2>::ones([b, n_atoms], &device);
        let ref_space_uid = Tensor::<B, 2, Int>::zeros([b, n_atoms], &device);
        let atom_to_token = Tensor::<B, 3>::zeros([b, n_atoms, n], &device);

        let cond = model.forward_diffusion_conditioning(
            s_trunk.clone(),
            z_trunk.clone(),
            rel_enc,
            ref_pos,
            ref_charge,
            ref_element,
            atom_pad_mask.clone(),
            ref_space_uid,
            atom_to_token.clone(),
            None,
        );

        let diffusion = model.forward_diffusion_sample(
            s_inputs,
            s_trunk,
            &cond,
            pad,
            atom_pad_mask,
            atom_to_token,
            Some(1),
            1,
            Some(SteeringParams::fast_path()),
        );
        assert_eq!(diffusion.sample_atom_coords.dims(), [1, n_atoms, 3]);

        let pdistogram = model.forward_distogram(z_trunk);
        assert_eq!(pdistogram.dims(), [b, n, n, 1, num_bins]);
    }
}
