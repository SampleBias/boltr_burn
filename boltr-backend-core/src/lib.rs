//! Shared backend-agnostic types for Boltz2 inference.
//!
//! Extracted from [`boltr-backend-tch`](https://github.com/SampleBias/Boltr) for use by
//! Burn and (during migration) LibTorch backends.

pub mod backend;
pub mod boltz_hparams;
pub mod checkpoint;
pub mod inference_keys;
pub mod predict_args;

pub use backend::{BackendFloatTensor, BackendIntTensor, Boltz2Backend, Boltz2ModelDims, PredictStepFeats, PredictStepOutput};
pub use boltz_hparams::Boltz2Hparams;
pub use checkpoint::{
    expected_keys_missing_in_safetensors, list_safetensor_names, safetensor_names_not_in_expected,
};
pub use inference_keys::{
    partition_safetensors_keys_for_inference, BOLTZ2_INFERENCE_TOP_LEVEL_KEYS,
};
pub use predict_args::{
    merge_predict_args_from_json, resolve_predict_args, Boltz2PredictArgs, PredictArgsCliOverrides,
};
