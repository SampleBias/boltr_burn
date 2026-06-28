//! Typed inference knobs (`predict_args`) aligned with Python `Boltz2.forward` / `predict_step`.
//!
//! **Precedence** (narrowest wins): CLI flags → YAML `predict_args` (when present) → checkpoint
//! `hyper_parameters["predict_args"]` JSON → [`Boltz2PredictArgs::default`] (Boltz-style inference defaults).

use serde::Serialize;
use serde_json::Value;

/// Inference-time steps and sample counts (Python: `recycling_steps`, `num_sampling_steps`,
/// `diffusion_samples` as `multiplicity`, `max_parallel_samples`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Boltz2PredictArgs {
    /// Trunk recycling iterations: loop runs `0..=recycling_steps` (Python `for i in range(recycling_steps + 1)`).
    pub recycling_steps: i64,
    /// Diffusion sampler steps; `None` defers to model default `num_sampling_steps`.
    pub sampling_steps: Option<i64>,
    /// Structure samples per batch (`multiplicity` in `AtomDiffusion::sample`).
    pub diffusion_samples: i64,
    /// Optional cap for parallel diffusion samples (steering / potentials path).
    pub max_parallel_samples: Option<i64>,
}

impl Default for Boltz2PredictArgs {
    /// Matches Python `Boltz2.forward` defaults (`boltz-reference/.../boltz1.py` / `boltz2.py`):
    /// `recycling_steps=0`, `num_sampling_steps=None`, `diffusion_samples=1`, `max_parallel_samples=None`.
    fn default() -> Self {
        Self {
            recycling_steps: 0,
            sampling_steps: None,
            diffusion_samples: 1,
            max_parallel_samples: None,
        }
    }
}

impl Boltz2PredictArgs {
    /// Quality-oriented inference values matching common Boltz prediction settings.
    #[must_use]
    pub fn quality_preset() -> Self {
        Self {
            recycling_steps: 3,
            sampling_steps: Some(200),
            diffusion_samples: 2,
            max_parallel_samples: Some(1),
        }
    }
}

/// CLI-only overrides (unset = do not override the merged checkpoint/YAML value).
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct PredictArgsCliOverrides {
    pub recycling_steps: Option<i64>,
    pub sampling_steps: Option<i64>,
    pub diffusion_samples: Option<i64>,
    pub max_parallel_samples: Option<i64>,
}

/// Merge keys from a JSON object into `out`. Recognized keys:
/// - `recycling_steps`
/// - `num_sampling_steps` or `sampling_steps`
/// - `diffusion_samples`
/// - `max_parallel_samples`
pub fn merge_predict_args_from_json(out: &mut Boltz2PredictArgs, v: &Value) {
    let Some(obj) = v.as_object() else {
        return;
    };
    if let Some(n) = obj.get("recycling_steps").and_then(Value::as_i64) {
        out.recycling_steps = n;
    }
    if let Some(n) = obj
        .get("num_sampling_steps")
        .or_else(|| obj.get("sampling_steps"))
        .and_then(Value::as_i64)
    {
        out.sampling_steps = Some(n);
    }
    if let Some(n) = obj.get("diffusion_samples").and_then(Value::as_i64) {
        out.diffusion_samples = n;
    }
    if let Some(n) = obj.get("max_parallel_samples").and_then(Value::as_i64) {
        out.max_parallel_samples = Some(n);
    }
}

impl crate::boltz_hparams::Boltz2Hparams {
    /// Resolve from checkpoint `predict_args` JSON only (defaults for missing keys).
    #[must_use]
    pub fn resolved_predict_args(&self) -> Boltz2PredictArgs {
        let mut p = Boltz2PredictArgs::default();
        if let Some(j) = &self.predict_args {
            merge_predict_args_from_json(&mut p, j);
        }
        p
    }
}

/// Full resolution: **CLI > YAML > checkpoint > defaults**.
///
/// `yaml_predict` is an optional JSON object (e.g. `predict_args` section from an input YAML).
#[must_use]
pub fn resolve_predict_args(
    hparams: &crate::boltz_hparams::Boltz2Hparams,
    yaml_predict: Option<&Value>,
    cli: PredictArgsCliOverrides,
) -> Boltz2PredictArgs {
    let mut p = hparams.resolved_predict_args();
    if let Some(y) = yaml_predict {
        merge_predict_args_from_json(&mut p, y);
    }
    if let Some(n) = cli.recycling_steps {
        p.recycling_steps = n;
    }
    if let Some(n) = cli.sampling_steps {
        p.sampling_steps = Some(n);
    }
    if let Some(n) = cli.diffusion_samples {
        p.diffusion_samples = n;
    }
    if let Some(n) = cli.max_parallel_samples {
        p.max_parallel_samples = Some(n);
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boltz_hparams::Boltz2Hparams;

    #[test]
    fn checkpoint_json_parses() {
        let j = serde_json::json!({
            "recycling_steps": 3,
            "num_sampling_steps": 200,
            "diffusion_samples": 2,
            "max_parallel_samples": 4
        });
        let mut p = Boltz2PredictArgs::default();
        merge_predict_args_from_json(&mut p, &j);
        assert_eq!(p.recycling_steps, 3);
        assert_eq!(p.sampling_steps, Some(200));
        assert_eq!(p.diffusion_samples, 2);
        assert_eq!(p.max_parallel_samples, Some(4));
    }

    #[test]
    fn quality_preset_uses_boltz_like_values() {
        let p = Boltz2PredictArgs::quality_preset();
        assert_eq!(p.recycling_steps, 3);
        assert_eq!(p.sampling_steps, Some(200));
        assert_eq!(p.diffusion_samples, 2);
        assert_eq!(p.max_parallel_samples, Some(1));
    }

    #[test]
    fn cli_overrides_checkpoint() {
        let h = Boltz2Hparams {
            predict_args: Some(serde_json::json!({ "recycling_steps": 1 })),
            ..Default::default()
        };
        let p = resolve_predict_args(
            &h,
            None,
            PredictArgsCliOverrides {
                recycling_steps: Some(5),
                ..Default::default()
            },
        );
        assert_eq!(p.recycling_steps, 5);
    }

    #[test]
    fn yaml_overrides_checkpoint_then_cli() {
        let h = Boltz2Hparams {
            predict_args: Some(serde_json::json!({ "recycling_steps": 1 })),
            ..Default::default()
        };
        let yaml = serde_json::json!({ "recycling_steps": 2 });
        let p = resolve_predict_args(&h, Some(&yaml), PredictArgsCliOverrides::default());
        assert_eq!(p.recycling_steps, 2);
    }
}
