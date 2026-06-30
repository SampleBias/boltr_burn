# trunk_init_golden.safetensors

Opt-in parity for `rel_pos` and `s_init` vs PyTorch (see [`scripts/export_trunk_init_golden.py`](../../../../scripts/export_trunk_init_golden.py)).

The file is **not** checked in until generated (requires `torch` + Boltz on `PYTHONPATH`).

```bash
PYTHONPATH=boltz-reference/src python3 scripts/export_trunk_init_golden.py
```

Rust: `BOLTR_RUN_TRUNK_INIT_GOLDEN=1 scripts/cargo-tch test -p boltr-backend-tch --features tch-backend trunk_init_allclose_python_golden`
