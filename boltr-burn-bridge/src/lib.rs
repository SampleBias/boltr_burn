//! Convert `boltr-io` collated batches into Burn tensors for predict-step.
//!
//! Mirrors `boltr-cli/src/collate_predict_bridge.rs` (tch path) — see Phase 1 in
//! `docs/IMPLEMENTATION_PLAN.md`.

use anyhow::{bail, Result};
use boltr_backend_burn::BurnPredictStepFeats;
use boltr_io::feature_batch::FeatureBatch;

/// Bridge a collated [`FeatureBatch`] into backend-specific predict-step features.
///
/// **Phase 0:** Token count only. Full tensor conversion lands with trunk parity work.
pub fn feature_batch_to_burn(_batch: &FeatureBatch) -> Result<BurnPredictStepFeats> {
    bail!("FeatureBatch → Burn bridge not yet implemented — see Phase 1")
}
