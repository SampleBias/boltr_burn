//! Partial Boltz2 hyperparameters from Lightning checkpoints (`hyper_parameters` dict).
//!
//! Export JSON with [`scripts/export_hparams_from_ckpt.py`](../../scripts/export_hparams_from_ckpt.py).
//!
//! Typed fields mirror common top-level keys in [`Boltz2`](../../../boltz-reference/src/boltz/model/models/boltz2.py);
//! anything else is preserved in [`Boltz2Hparams::other`].

use anyhow::{Context, Result};
use serde::Deserialize;

/// Subset of keys used to size the Rust inference graph; nested dicts kept as JSON.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct Boltz2Hparams {
    #[serde(default)]
    pub atom_s: Option<i64>,
    #[serde(default)]
    pub atom_z: Option<i64>,
    #[serde(default)]
    pub token_s: Option<i64>,
    #[serde(default)]
    pub token_z: Option<i64>,
    #[serde(default)]
    pub num_bins: Option<i64>,
    #[serde(default)]
    pub num_blocks: Option<i64>,
    /// Matches Python `Boltz2(bond_type_feature=…)` when present in checkpoint hparams.
    #[serde(default)]
    pub bond_type_feature: Option<bool>,
    #[serde(default)]
    pub pairformer_args: Option<PairformerArgs>,
    #[serde(default)]
    pub embedder_args: Option<serde_json::Value>,
    #[serde(default)]
    pub msa_args: Option<serde_json::Value>,
    #[serde(default)]
    pub training_args: Option<serde_json::Value>,
    #[serde(default)]
    pub validation_args: Option<serde_json::Value>,
    #[serde(default)]
    pub score_model_args: Option<serde_json::Value>,
    #[serde(default)]
    pub diffusion_process_args: Option<serde_json::Value>,
    #[serde(default)]
    pub diffusion_loss_args: Option<serde_json::Value>,
    #[serde(default)]
    pub confidence_model_args: Option<serde_json::Value>,
    #[serde(default)]
    pub affinity_model_args: Option<serde_json::Value>,
    #[serde(default)]
    pub template_args: Option<serde_json::Value>,
    #[serde(default)]
    pub predict_args: Option<serde_json::Value>,
    #[serde(default)]
    pub steering_args: Option<serde_json::Value>,
    /// Top-level Lightning flag (sizing affinity head in checkpoint).
    #[serde(default)]
    pub confidence_prediction: Option<bool>,
    #[serde(default)]
    pub affinity_prediction: Option<bool>,
    #[serde(default)]
    pub affinity_mw_correction: Option<bool>,
    /// Atom window sizes (Boltz2 `atoms_per_window_queries` / `atoms_per_window_keys`).
    #[serde(default)]
    pub atoms_per_window_queries: Option<i64>,
    #[serde(default)]
    pub atoms_per_window_keys: Option<i64>,
    /// Remaining Lightning `hyper_parameters` keys (flags, optimizers, etc.).
    #[serde(flatten)]
    pub other: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct PairformerArgs {
    #[serde(default)]
    pub token_s: Option<i64>,
    #[serde(default)]
    pub token_z: Option<i64>,
    #[serde(default)]
    pub num_blocks: Option<i64>,
}

/// Lightning checkpoints sometimes store nested dicts as Python `repr()` strings; `json.dump(..., default=str)`
/// then embeds those as JSON strings. Coerce dict/array-looking strings back to objects before serde runs.
fn parse_json_or_python_literal(s: &str) -> Result<serde_json::Value> {
    let t = s.trim();
    if t.is_empty() {
        anyhow::bail!("empty");
    }
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(t) {
        return Ok(v);
    }
    let normalized = t
        .replace("False", "false")
        .replace("True", "true")
        .replace("None", "null");
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&normalized) {
        return Ok(v);
    }
    json5::from_str::<serde_json::Value>(&normalized).context("json5 parse of Python-style literal")
}

fn normalize_stringified_nested_dicts(v: &mut serde_json::Value) -> Result<()> {
    let Some(obj) = v.as_object_mut() else {
        return Ok(());
    };
    let keys: Vec<String> = obj.keys().cloned().collect();
    for key in keys {
        let Some(entry) = obj.get_mut(&key) else {
            continue;
        };
        let serde_json::Value::String(s) = entry.clone() else {
            continue;
        };
        let trimmed = s.trim();
        if !(trimmed.starts_with('{') || trimmed.starts_with('[')) {
            continue;
        }
        match parse_json_or_python_literal(trimmed) {
            Ok(parsed) => {
                *entry = parsed;
            }
            Err(e) => {
                if key == "pairformer_args" {
                    return Err(e).context(format!(
                        "pairformer_args is a string that could not be parsed as JSON/object: {}",
                        trimmed.chars().take(120).collect::<String>()
                    ));
                }
                tracing::warn!(
                    key = %key,
                    error = %e,
                    "boltz2_hparams: could not parse stringified nested value; keeping string"
                );
            }
        }
    }
    Ok(())
}

impl Boltz2Hparams {
    /// Parse from JSON bytes (exported by the Python script).
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self> {
        let mut v: serde_json::Value = serde_json::from_slice(bytes)?;
        normalize_stringified_nested_dicts(&mut v)?;
        Ok(serde_json::from_value(v)?)
    }

    /// Alias for [`Self::from_json_slice`]: loads the full Lightning `hyper_parameters` object
    /// written by [`scripts/export_hparams_from_ckpt.py`](../../scripts/export_hparams_from_ckpt.py).
    pub fn from_lightning_hyper_parameters_json(bytes: &[u8]) -> Result<Self> {
        Self::from_json_slice(bytes)
    }

    #[must_use]
    pub fn resolved_token_s(&self) -> i64 {
        self.token_s
            .or_else(|| self.pairformer_args.as_ref().and_then(|p| p.token_s))
            .unwrap_or(384)
    }

    #[must_use]
    pub fn resolved_token_z(&self) -> i64 {
        self.token_z
            .or_else(|| self.pairformer_args.as_ref().and_then(|p| p.token_z))
            .unwrap_or(128)
    }

    #[must_use]
    pub fn resolved_num_pairformer_blocks(&self) -> Option<i64> {
        self.num_blocks
            .or_else(|| self.pairformer_args.as_ref().and_then(|p| p.num_blocks))
    }

    #[must_use]
    pub fn resolved_bond_type_feature(&self) -> bool {
        self.bond_type_feature.unwrap_or(false)
    }

    /// `atoms_per_window_queries` from explicit JSON, `other`, or default `32`.
    #[must_use]
    pub fn resolved_atoms_per_window_queries(&self) -> i64 {
        self.atoms_per_window_queries
            .or_else(|| {
                self.other
                    .get("atoms_per_window_queries")
                    .and_then(|v| v.as_i64())
            })
            .unwrap_or(32)
    }

    /// `atoms_per_window_keys` from explicit JSON, `other`, or default `128`.
    #[must_use]
    pub fn resolved_atoms_per_window_keys(&self) -> i64 {
        self.atoms_per_window_keys
            .or_else(|| {
                self.other
                    .get("atoms_per_window_keys")
                    .and_then(|v| v.as_i64())
            })
            .unwrap_or(128)
    }

    /// `training_args["recycling_steps"]` when present (Boltz2 training / finetune scripts).
    #[must_use]
    pub fn recycling_steps_from_training_args(&self) -> Option<i64> {
        self.training_args
            .as_ref()
            .and_then(|v| v.get("recycling_steps"))
            .and_then(|x| x.as_i64())
    }

    /// Count of extra top-level keys stored in [`Self::other`].
    #[must_use]
    pub fn other_key_count(&self) -> usize {
        self.other.len()
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn parses_minimal_json() {
        let j = br#"{"token_s": 384, "token_z": 128, "num_blocks": 4}"#;
        let h = Boltz2Hparams::from_json_slice(j).unwrap();
        assert_eq!(h.resolved_token_s(), 384);
        assert_eq!(h.resolved_token_z(), 128);
        assert_eq!(h.resolved_num_pairformer_blocks(), Some(4));
        assert!(!h.resolved_bond_type_feature());
        assert_eq!(h.other_key_count(), 0);
    }

    #[test]
    fn parses_bond_type_feature() {
        let j = br#"{"token_s": 384, "token_z": 128, "num_blocks": 4, "bond_type_feature": true}"#;
        let h = Boltz2Hparams::from_json_slice(j).unwrap();
        assert!(h.resolved_bond_type_feature());
    }

    #[test]
    fn parses_committed_minimal_fixture() {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hparams/minimal.json");
        let raw = std::fs::read(&p).expect("minimal.json");
        let h = Boltz2Hparams::from_json_slice(&raw).unwrap();
        assert_eq!(h.resolved_token_s(), 384);
        assert_eq!(h.resolved_token_z(), 128);
    }

    #[test]
    fn preserves_unknown_top_level_in_other() {
        let j = br#"{"token_s": 384, "token_z": 128, "num_blocks": 4, "ema": true}"#;
        let h = Boltz2Hparams::from_json_slice(j).unwrap();
        assert_eq!(h.other.get("ema"), Some(&serde_json::json!(true)));
    }

    #[test]
    fn resolved_atoms_per_window_from_other_map() {
        let mut h = Boltz2Hparams::default();
        h.other.insert(
            "atoms_per_window_queries".to_string(),
            serde_json::json!(40),
        );
        h.other
            .insert("atoms_per_window_keys".to_string(), serde_json::json!(80));
        assert_eq!(h.resolved_atoms_per_window_queries(), 40);
        assert_eq!(h.resolved_atoms_per_window_keys(), 80);
    }

    #[test]
    fn resolved_atoms_per_window_top_level_json() {
        let j = br#"{"token_s": 384, "atoms_per_window_queries": 16, "atoms_per_window_keys": 64}"#;
        let h = Boltz2Hparams::from_json_slice(j).unwrap();
        assert_eq!(h.resolved_atoms_per_window_queries(), 16);
        assert_eq!(h.resolved_atoms_per_window_keys(), 64);
    }

    #[test]
    fn nested_training_args_and_recycling_steps() {
        let j = br#"{"token_s": 384, "token_z": 128, "training_args": {"recycling_steps": 3, "max_lr": 0.001}}"#;
        let h = Boltz2Hparams::from_json_slice(j).unwrap();
        assert_eq!(h.recycling_steps_from_training_args(), Some(3));
        assert!(h.training_args.as_ref().unwrap().get("max_lr").is_some());
    }

    #[test]
    fn parses_sample_full_fixture() {
        let p =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hparams/sample_full.json");
        let raw = std::fs::read(&p).expect("sample_full.json");
        let h = Boltz2Hparams::from_json_slice(&raw).unwrap();
        assert_eq!(h.atom_s, Some(128));
        assert_eq!(h.resolved_token_s(), 384);
        assert!(h.embedder_args.is_some());
        assert!(h.other.contains_key("use_templates"));
    }

    #[test]
    fn from_lightning_alias_matches_from_json_slice() {
        let j = br#"{"token_s": 384, "token_z": 128, "num_blocks": 4}"#;
        let a = Boltz2Hparams::from_json_slice(j).unwrap();
        let b = Boltz2Hparams::from_lightning_hyper_parameters_json(j).unwrap();
        assert_eq!(a.resolved_token_s(), b.resolved_token_s());
    }

    /// `json.dump(..., default=str)` can turn nested dicts into Python repr strings inside JSON.
    #[test]
    fn parses_pairformer_args_embedded_as_python_repr_string() {
        let inner = "{'num_blocks': 64, 'num_heads': 16, 'dropout': 0.25, 'post_layer_norm': False, 'activation_checkpointing': True, 'use_trifast': True}";
        let j = serde_json::json!({
            "token_s": 384,
            "pairformer_args": inner
        })
        .to_string();
        let h = Boltz2Hparams::from_json_slice(j.as_bytes()).expect("coerce + parse");
        assert_eq!(h.resolved_num_pairformer_blocks(), Some(64));
    }
}
