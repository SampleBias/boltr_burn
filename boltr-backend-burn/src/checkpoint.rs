//! Load f32 tensors and module weights from safetensors for Burn backends.

use std::path::Path;

use anyhow::{Context, Result};
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Device, ElementConversion, Tensor, TensorData};
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use safetensors::SafeTensors;

/// Load one f32 tensor by name into a Burn tensor of the given rank.
pub fn load_f32_tensor<B: Backend, const D: usize>(
    path: &Path,
    name: &str,
    device: &Device<B>,
) -> Result<Tensor<B, D>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let st = SafeTensors::deserialize(&bytes).context("parse safetensors")?;
    let view = st
        .tensor(name)
        .with_context(|| format!("missing tensor {name:?}"))?;
    let shape: Vec<usize> = view.shape().to_vec();
    anyhow::ensure!(
        shape.len() == D,
        "tensor {name} rank {} != expected {D}",
        shape.len()
    );
    let data: Vec<f32> = view
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok(Tensor::from_data(TensorData::new(data, shape), device))
}

/// Load a module record from safetensors (keys must match Burn record paths).
pub fn load_module_from_safetensors<M, B>(
    mut module: M,
    path: &Path,
    _device: &Device<B>,
) -> Result<M>
where
    M: Module<B> + ModuleSnapshot<B>,
    B: Backend,
{
    let mut store = SafetensorsStore::from_file(path).with_from_adapter(PyTorchToBurnAdapter);
    module
        .load_from(&mut store)
        .with_context(|| format!("load safetensors into module {}", path.display()))?;
    Ok(module)
}

/// Golden-test helper: max absolute difference between two tensors.
pub fn max_abs_diff<B: Backend, const D: usize>(a: &Tensor<B, D>, b: &Tensor<B, D>) -> f64 {
    (a.clone() - b.clone())
        .abs()
        .max()
        .into_scalar()
        .elem::<f64>()
}

/// Assert allclose with rtol/atol (panics with message on failure).
pub fn assert_allclose<B: Backend, const D: usize>(
    name: &str,
    got: &Tensor<B, D>,
    expected: &Tensor<B, D>,
    rtol: f64,
    atol: f64,
) {
    let diff = max_abs_diff(got, expected);
    let scale = expected
        .clone()
        .abs()
        .max()
        .into_scalar()
        .elem::<f64>()
        .max(1.0);
    assert!(
        diff < atol + rtol * scale,
        "{name} golden mismatch: max_abs_diff={diff} (rtol={rtol} atol={atol} scale={scale})"
    );
}
