//! Device selection and runtime probes for Burn backends.

use burn::backend::NdArray;
use burn::tensor::Device;
use serde::Serialize;

/// Runtime probe results for `boltr-burn doctor`.
#[derive(Debug, Clone, Serialize)]
pub struct BackendProbe {
    pub default_feature: &'static str,
    pub ndarray_available: bool,
    #[cfg(feature = "burn-cuda")]
    pub cuda_compiled: bool,
    #[cfg(not(feature = "burn-cuda"))]
    pub cuda_compiled: bool,
    #[cfg(feature = "burn-wgpu")]
    pub wgpu_compiled: bool,
    #[cfg(not(feature = "burn-wgpu"))]
    pub wgpu_compiled: bool,
}

/// Which Burn backend feature set was compiled into this binary.
#[must_use]
pub fn compiled_default_backend() -> &'static str {
    if cfg!(feature = "burn-cuda") {
        "burn-cuda"
    } else if cfg!(feature = "burn-wgpu") {
        "burn-wgpu"
    } else {
        "burn-ndarray"
    }
}

/// Probe compiled backends (does not guarantee GPU runtime availability).
#[must_use]
pub fn probe_backends() -> BackendProbe {
    BackendProbe {
        default_feature: compiled_default_backend(),
        ndarray_available: cfg!(feature = "burn-ndarray"),
        cuda_compiled: cfg!(feature = "burn-cuda"),
        wgpu_compiled: cfg!(feature = "burn-wgpu"),
    }
}

/// Parse `--device` strings: `cpu`, `cuda`, `cuda:N`, `wgpu`.
pub fn parse_device_spec(spec: &str) -> Result<DeviceSpec, String> {
    let s = spec.trim().to_lowercase();
    if s == "cpu" || s == "ndarray" {
        return Ok(DeviceSpec::Cpu);
    }
    if s == "cuda" || s.starts_with("cuda:") {
        if !cfg!(feature = "burn-cuda") {
            return Err("this binary was not built with --features burn-cuda".into());
        }
        let index = if s == "cuda" {
            0
        } else {
            s.strip_prefix("cuda:")
                .and_then(|n| n.parse().ok())
                .ok_or_else(|| format!("invalid cuda device spec: {spec}"))?
        };
        return Ok(DeviceSpec::Cuda(index));
    }
    if s == "wgpu" {
        if !cfg!(feature = "burn-wgpu") {
            return Err("this binary was not built with --features burn-wgpu".into());
        }
        return Ok(DeviceSpec::Wgpu);
    }
    Err(format!("unknown device spec: {spec} (expected cpu, cuda[:N], wgpu)"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceSpec {
    Cpu,
    Cuda(usize),
    Wgpu,
}

/// Default device for the compiled backend (CPU ndarray fallback).
pub fn default_device() -> Device<NdArray> {
    Default::default()
}
