# Boltz2 smoke safetensors (strict load)

This directory holds a **small** pinned [`Boltz2Model::with_options`](../../src/boltz2/model.rs) export (`token_s=64`, `token_z=32`, one pairformer block, no bond-type embedding) so tests can call `load_from_safetensors_require_all_vars` without a full Lightning checkpoint.

Regenerate after changing the Rust graph (new `VarStore` names):

```bash
scripts/cargo-tch run -p boltr-backend-tch --bin gen_boltz2_smoke_safetensors --features tch-backend
```

For **real** Boltz2 checkpoint alignment, use [`verify_boltz2_safetensors`](../../src/bin/verify_boltz2_safetensors.rs); see [DEVELOPMENT.md](../../../../DEVELOPMENT.md).
