# Pairformer layer numerical golden

`PairformerLayer` (Boltz2 `AttentionPairBiasV2`, triangle mult/attn fallback, transitions) vs Python with `use_kernels=False` and `chunk_size_tri_attn=None`.

## Regenerate fixture

From repo root (venv with `torch`, `safetensors`, `numpy`, …):

```bash
PYTHONPATH=boltz-reference/src python scripts/export_pairformer_golden.py
```

Writes `pairformer_layer_golden.safetensors` (weights under `layers.0.*` + `golden.*` tensors).

## Run Rust check (opt-in)

```bash
BOLTR_RUN_PAIRFORMER_GOLDEN=1 LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH" \
  scripts/cargo-tch test -p boltr-backend-tch --features tch-backend pairformer_layer_allclose_python_golden
```

Default `cargo test` skips the assertion so clones without LibTorch or without regenerating the file stay green.
