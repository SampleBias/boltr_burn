//! Opt-in pairformer layer golden vs Python export.

use std::path::Path;

use boltr_backend_burn::checkpoint::{assert_allclose, load_f32_tensor, load_module_from_safetensors};
use boltr_backend_burn::PairformerLayer;
use burn::backend::NdArray;
use burn::module::Module;

type B = NdArray;

fn fixture_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pairformer_golden/pairformer_layer_golden.safetensors")
}

fn pairformer_golden_requested() -> bool {
    std::env::var("BOLTR_RUN_PAIRFORMER_GOLDEN")
        .map(|v| v == "1")
        .unwrap_or(false)
}

#[derive(Module, Debug)]
struct LayerWrapper<B: burn::tensor::backend::Backend> {
    layers: Vec<PairformerLayer<B>>,
}

#[test]
fn pairformer_layer_allclose_python_golden() {
    if !pairformer_golden_requested() {
        return;
    }
    let path = fixture_path();
    assert!(
        path.is_file(),
        "missing {}; run scripts/export_pairformer_golden.py in Boltr",
        path.display()
    );

    let device = Default::default();
    let token_s = 32;
    let token_z = 24;
    let wrapper = LayerWrapper {
        layers: vec![PairformerLayer::<B>::new(
            &device, token_s, token_z, 4, 0.0, 32, 4, false,
        )],
    };
    let wrapper = load_module_from_safetensors(wrapper, &path, &device)
        .unwrap_or_else(|e| panic!("load weights {}: {e}", path.display()));

    let s = load_f32_tensor::<B, 3>(&path, "golden.in_s", &device).unwrap();
    let z = load_f32_tensor::<B, 4>(&path, "golden.in_z", &device).unwrap();
    let mask = load_f32_tensor::<B, 3>(&path, "golden.mask", &device).unwrap();
    let pair_mask = load_f32_tensor::<B, 3>(&path, "golden.pair_mask", &device).unwrap();
    let s_ref = load_f32_tensor::<B, 3>(&path, "golden.s_out", &device).unwrap();
    let z_ref = load_f32_tensor::<B, 4>(&path, "golden.z_out", &device).unwrap();

    let (s_rust, z_rust) = wrapper.layers[0].forward(
        s, z, mask, pair_mask, None, false, false,
    );

    assert_allclose("s_out", &s_rust, &s_ref, 1e-4, 1e-5);
    assert_allclose("z_out", &z_rust, &z_ref, 1e-4, 1e-5);
}
