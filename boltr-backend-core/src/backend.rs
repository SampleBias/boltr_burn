//! Backend-agnostic trait for Boltz2 inference runtimes (Burn, tch, etc.).

use anyhow::Result;

use crate::boltz_hparams::Boltz2Hparams;
use crate::predict_args::Boltz2PredictArgs;

/// Marker for backend-specific float tensors in predict-step I/O.
pub trait BackendFloatTensor: Send + Sync {}

/// Marker for backend-specific integer tensors in predict-step I/O.
pub trait BackendIntTensor: Send + Sync {}

/// Predict-step feature batch (backend-specific tensor handles).
pub trait PredictStepFeats<B: Boltz2Backend> {
    fn token_count(&self) -> usize;
}

/// Predict-step outputs consumed by `boltr-io` writers.
pub trait PredictStepOutput<B: Boltz2Backend> {
    fn sample_count(&self) -> usize;
}

/// Backend-agnostic Boltz2 inference surface.
///
/// Implementations:
/// - `BurnBackend<B>` — Burn + CubeCL (this repo)
/// - `TchBackend` — existing `boltr-backend-tch` (A/B during migration)
pub trait Boltz2Backend: Sized {
    type Device: Clone + Send + Sync + std::fmt::Debug;
    type FloatTensor: BackendFloatTensor;
    type IntTensor: BackendIntTensor;
    type Model: Send + Sync;
    type Feats: PredictStepFeats<Self>;
    type Output: PredictStepOutput<Self>;

    fn backend_name() -> &'static str;

    fn predict_step(
        model: &Self::Model,
        feats: Self::Feats,
        args: &Boltz2PredictArgs,
        hparams: &Boltz2Hparams,
    ) -> Result<Self::Output>;
}

/// Resolved model dimensions shared across backends.
#[derive(Debug, Clone)]
pub struct Boltz2ModelDims {
    pub token_s: i64,
    pub token_z: i64,
    pub num_pairformer_blocks: i64,
    pub bond_type_feature: bool,
}

impl Boltz2ModelDims {
    #[must_use]
    pub fn from_hparams(h: &Boltz2Hparams) -> Self {
        Self {
            token_s: h.resolved_token_s(),
            token_z: h.resolved_token_z(),
            num_pairformer_blocks: h.resolved_num_pairformer_blocks().unwrap_or(4),
            bond_type_feature: h.resolved_bond_type_feature(),
        }
    }
}
