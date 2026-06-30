//! Burn ML backend for Boltz-2 inference.
//!
//! Module layout mirrors [`boltr-backend-tch`](https://github.com/SampleBias/Boltr) — see
//! `docs/IMPLEMENTATION_PLAN.md` for the port checklist.

pub mod attention;
pub mod backend;
pub mod boltz2;
pub mod burn_compat;
pub mod checkpoint;
pub mod device;
pub mod layers;
pub mod tensor_ops;

pub use attention::AttentionPairBiasV2;
pub use backend::{BurnNdArrayBackend, BurnPredictStepFeats, BurnPredictStepOutput, NdArrayBackend};
pub use boltz2::model::{Boltz2BurnModel, Boltz2BurnModelConfig, Boltz2DiffusionArgs};
pub use boltz2::{
    apply_affinity_mw_correction, AffinityModule, AffinityModuleConfig, AffinityOutput,
    AtomDiffusion, AtomDiffusionConfig, AtomEncoderBatchFeats, AtomEncoderFlags,
    BFactorModule, ConfidenceModule, ConfidenceModuleConfig, ConfidenceOutput,
    DiffusionConditioning, DiffusionConditioningOutput, DiffusionSampleOutput, DistogramModule,
    InputEmbedder, MsaFeatures, MsaModule, RelPosFeatures, RelativePositionEncoder,
    SteeringParams, TemplateFeatures, TemplateV2Module, TrunkV2, BOLTZ_MSA_PROFILE_IN,
    BOLTZ_NUM_TOKENS,
};
pub use device::{default_device, parse_device_spec, probe_backends, BackendProbe};
pub use layers::{
    OuterProductMeanMsa, PairWeightedAveraging, PairformerLayer, PairformerModule,
    PairformerNoSeqLayer, PairformerNoSeqModule, Transition, TriangleAttention,
    TriangleAttentionStartingNode, TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing,
};
