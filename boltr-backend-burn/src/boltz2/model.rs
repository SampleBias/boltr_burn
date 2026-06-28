//! Boltz2 high-level model skeleton on Burn `Module` types.
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
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Device;

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
        let mut h = Boltz2Hparams::default();
        h.token_s = Some(token_s);
        h.token_z = Some(token_z);
        h.num_blocks = num_blocks;
        Self::from_hparams(&h)
    }
}

/// Phase 0 skeleton — root fields mirror Lightning `state_dict` top-level names.
///
/// Submodule ports (pairformer, MSA, diffusion, heads) land in Phases 1–2.
#[derive(Module, Debug)]
pub struct Boltz2BurnModel<B: Backend> {
    pub s_init: Linear<B>,
    pub z_init_1: Linear<B>,
    pub z_init_2: Linear<B>,
    pub s_norm: LayerNorm<B>,
    pub z_norm: LayerNorm<B>,
    pub s_recycle: Linear<B>,
    pub z_recycle: Linear<B>,
    pub token_bonds: Linear<B>,
    pub token_bonds_type: Option<Embedding<B>>,
}

impl<B: Backend> Boltz2BurnModel<B> {
    pub fn new(device: &Device<B>, config: &Boltz2BurnModelConfig) -> Self {
        let token_s = config.dims.token_s as usize;
        let token_z = config.dims.token_z as usize;

        Self {
            s_init: LinearConfig::new(token_s, token_s)
                .with_bias(false)
                .init(device),
            z_init_1: LinearConfig::new(token_s, token_z)
                .with_bias(false)
                .init(device),
            z_init_2: LinearConfig::new(token_s, token_z)
                .with_bias(false)
                .init(device),
            s_norm: LayerNormConfig::new(token_s).init(device),
            z_norm: LayerNormConfig::new(token_z).init(device),
            s_recycle: LinearConfig::new(token_s, token_s).init(device),
            z_recycle: LinearConfig::new(token_z, token_z).init(device),
            token_bonds: LinearConfig::new(1, token_z)
                .with_bias(false)
                .init(device),
            token_bonds_type: config.dims.bond_type_feature.then(|| {
                EmbeddingConfig::new(7, token_z).init(device)
            }),
        }
    }

    /// Collect Burn parameter keys (Lightning `state_dict` paths).
    pub fn parameter_names(&self) -> Vec<String> {
        let mut names = vec![
            "s_init.weight".into(),
            "z_init_1.weight".into(),
            "z_init_2.weight".into(),
            "s_norm.weight".into(),
            "s_norm.bias".into(),
            "z_norm.weight".into(),
            "z_norm.bias".into(),
            "s_recycle.weight".into(),
            "s_recycle.bias".into(),
            "z_recycle.weight".into(),
            "z_recycle.bias".into(),
            "token_bonds.weight".into(),
        ];
        if self.token_bonds_type.is_some() {
            names.push("token_bonds_type.weight".into());
        }
        names.sort();
        names
    }

    /// Keys in the model graph absent from a safetensors file.
    pub fn keys_missing_in_safetensors(&self, path: &Path) -> Result<Vec<String>> {
        let names = self.parameter_names();
        expected_keys_missing_in_safetensors(path, &names)
    }

    /// Safetensors keys not mapped into this skeleton (naming mismatch or unported modules).
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
        Ok((Self::new(device, &config), config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn skeleton_registers_trunk_init_keys() {
        let device = Default::default();
        let config = Boltz2BurnModelConfig::with_defaults(384, 128, Some(4));
        let model = Boltz2BurnModel::<B>::new(&device, &config);
        let names = model.parameter_names();
        assert!(names.iter().any(|k| k.starts_with("s_init")));
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
}
