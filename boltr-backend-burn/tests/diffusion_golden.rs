//! Opt-in diffusion goldens (`BOLTR_RUN_DIFFUSION_GOLDEN=1`).
//!
//! Generate fixtures with `scripts/export_diffusion_golden.py` when the Python Boltz tree is available.
//! Default `cargo test` skips heavy assertions.

use std::path::Path;

fn diffusion_golden_requested() -> bool {
    std::env::var("BOLTR_RUN_DIFFUSION_GOLDEN")
        .map(|v| v == "1")
        .unwrap_or(false)
}

fn diffusion_fixture() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/diffusion_golden/diffusion_step_golden.safetensors")
}

#[test]
fn diffusion_step_golden_opt_in() {
    if !diffusion_golden_requested() {
        return;
    }
    let p = diffusion_fixture();
    assert!(
        p.is_file(),
        "missing {}; generate with scripts/export_diffusion_golden.py when present upstream",
        p.display()
    );
}
