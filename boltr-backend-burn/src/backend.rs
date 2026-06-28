//! Burn backend type aliases and [`Boltz2Backend`] implementation stub.

use anyhow::{bail, Result};
use burn::backend::NdArray;
use burn::tensor::Device;

use boltr_backend_core::{
    BackendFloatTensor, BackendIntTensor, Boltz2Backend, Boltz2Hparams, Boltz2PredictArgs,
    PredictStepFeats, PredictStepOutput,
};

pub type NdArrayBackend = NdArray;

/// Placeholder float tensor handle until predict-step I/O is wired.
#[derive(Debug, Clone)]
pub struct BurnFloatTensor;

impl BackendFloatTensor for BurnFloatTensor {}

/// Placeholder int tensor handle until predict-step I/O is wired.
#[derive(Debug, Clone)]
pub struct BurnIntTensor;

impl BackendIntTensor for BurnIntTensor {}

/// Stub predict-step features (Phase 1+).
#[derive(Debug)]
pub struct BurnPredictStepFeats {
    pub token_count: usize,
}

impl PredictStepFeats<BurnNdArrayBackend> for BurnPredictStepFeats {
    fn token_count(&self) -> usize {
        self.token_count
    }
}

/// Stub predict-step output (Phase 2+).
#[derive(Debug)]
pub struct BurnPredictStepOutput {
    pub sample_count: usize,
}

impl PredictStepOutput<BurnNdArrayBackend> for BurnPredictStepOutput {
    fn sample_count(&self) -> usize {
        self.sample_count
    }
}

/// Type alias for the default CPU backend implementation.
pub struct BurnNdArrayBackend;

impl Boltz2Backend for BurnNdArrayBackend {
    type Device = Device<NdArray>;
    type FloatTensor = BurnFloatTensor;
    type IntTensor = BurnIntTensor;
    type Model = crate::boltz2::model::Boltz2BurnModel<NdArray>;
    type Feats = BurnPredictStepFeats;
    type Output = BurnPredictStepOutput;

    fn backend_name() -> &'static str {
        "burn-ndarray"
    }

    fn predict_step(
        _model: &Self::Model,
        _feats: Self::Feats,
        _args: &Boltz2PredictArgs,
        _hparams: &Boltz2Hparams,
    ) -> Result<Self::Output> {
        bail!("predict_step not yet implemented — see Phase 1/2 in docs/IMPLEMENTATION_PLAN.md")
    }
}
