# Burn numerical tolerances

Burn backend golden tests follow [Boltr `docs/NUMERICAL_TOLERANCES.md`](https://github.com/SampleBias/Boltr/blob/main/docs/NUMERICAL_TOLERANCES.md) unless noted below.

| Test class | rtol | atol | Notes |
|------------|------|------|-------|
| Featurizer / collate (via `boltr-io`) | 1e-5 | 1e-6 | Unchanged — I/O layer |
| Module goldens (embedder, pairformer, MSA) | 1e-4 | 1e-5 | Same as tch backend |
| Trunk integration | 1e-4 | 1e-5 | CPU ndarray first |
| Full predict vs Boltz | per `regression_tol.env` | | Looser end-to-end |

## Burn-specific notes

- **CPU first:** Module goldens run on `burn-ndarray` before GPU backends to isolate graph bugs from kernel drift.
- **CubeCL / CUDA:** If GPU kernels diverge slightly from LibTorch, document justified tolerances here with fixture name and max delta observed.
- **Non-goal v1:** cuEquivariance fused triangle kernels — parity target is `use_kernels=False` PyTorch path.

## References

- [Boltr TENSOR_CONTRACT.md](https://github.com/SampleBias/Boltr/blob/main/docs/TENSOR_CONTRACT.md)
- [BOLTR_BURN_PRD_AND_SPEC.md](../BOLTR_BURN_PRD_AND_SPEC.md) §4.5
