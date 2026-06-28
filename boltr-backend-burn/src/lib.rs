//! Burn ML backend for Boltz-2 inference.
//!
//! Module layout mirrors [`boltr-backend-tch`](https://github.com/SampleBias/Boltr) — see
//! `BOLTR_BURN_PRD_AND_SPEC.md` for the port checklist.

pub mod backend;
pub mod boltz2;
pub mod device;

pub use backend::{BurnNdArrayBackend, BurnPredictStepFeats, BurnPredictStepOutput, NdArrayBackend};
pub use boltz2::model::{Boltz2BurnModel, Boltz2BurnModelConfig};
pub use device::{default_device, parse_device_spec, probe_backends, BackendProbe};
