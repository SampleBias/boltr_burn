//! Opt-in golden parity vs Python exports (mirrors boltr-backend-tch tests).

use std::path::Path;

use boltr_backend_burn::checkpoint::{assert_allclose, load_f32_tensor, load_module_from_safetensors};
use boltr_backend_burn::{RelPosFeatures, RelativePositionEncoder};
use burn::backend::NdArray;
use burn::module::Module;
use burn::record::Record;
use burn::tensor::{Int, Tensor};

type B = NdArray;

fn fixture_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/trunk_init_golden/trunk_init_golden.safetensors")
}

fn trunk_init_golden_requested() -> bool {
    std::env::var("BOLTR_RUN_TRUNK_INIT_GOLDEN")
        .map(|v| v == "1")
        .unwrap_or(false)
}

#[derive(Module, Debug)]
struct RelPosWrapper<B: burn::tensor::backend::Backend> {
    rel_pos: RelativePositionEncoder<B>,
}

#[test]
fn trunk_init_allclose_python_golden() {
    if !trunk_init_golden_requested() {
        return;
    }
    let path = fixture_path();
    assert!(
        path.is_file(),
        "missing {}; run scripts/export_trunk_init_golden.py in Boltr",
        path.display()
    );

    let device = Default::default();
    let token_s = 32_usize;
    let token_z = 24_usize;

    let wrapper = RelPosWrapper {
        rel_pos: RelativePositionEncoder::<B>::new(&device, token_z, None, None, false, false),
    };
    let wrapper = load_module_from_safetensors(wrapper, &path, &device)
        .unwrap_or_else(|e| panic!("load weights {}: {e}", path.display()));
    let rel = &wrapper.rel_pos;

    let b = 2;
    let n = 7;
    let asym_id = Tensor::<B, 2, Int>::zeros([b, n], &device);
    let residue_index =
        Tensor::<B, 1, Int>::arange(0..n as i64, &device).reshape([1, n]).repeat(&[b, 1]);
    let entity_id = Tensor::<B, 2, Int>::zeros([b, n], &device);
    let token_index = residue_index.clone();
    let sym_id = Tensor::<B, 2, Int>::zeros([b, n], &device);
    let cyclic_period = Tensor::<B, 2, Int>::zeros([b, n], &device);
    let rel_f = RelPosFeatures {
        asym_id: &asym_id,
        residue_index: &residue_index,
        entity_id: &entity_id,
        token_index: &token_index,
        sym_id: &sym_id,
        cyclic_period: &cyclic_period,
    };
    let rel_rust = rel.forward(rel_f);
    let rel_ref = load_f32_tensor::<B, 4>(&path, "golden.rel_pos_out", &device).unwrap();
    assert_allclose("rel_pos_out", &rel_rust, &rel_ref, 1e-4, 1e-5);

    let s_w = load_f32_tensor::<B, 2>(&path, "s_init.weight", &device).unwrap();
    let s_in = load_f32_tensor::<B, 3>(&path, "golden.s_in", &device).unwrap();
    let [b, n, d] = s_in.dims();
    let s_exp = s_in
        .reshape([b * n, d])
        .matmul(s_w.transpose())
        .reshape([b, n, d]);
    let s_ref = load_f32_tensor::<B, 3>(&path, "golden.s_init_out", &device).unwrap();
    assert_allclose("s_init_out", &s_exp, &s_ref, 1e-4, 1e-4);
}
