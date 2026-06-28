//! Top-level `state_dict` segments for the Rust Boltz2 inference graph (`boltz2/model.rs` `Boltz2Model`).
//! This module has **no** `tch` dependency — tests run on CI without LibTorch.

use std::collections::HashSet;

/// Top-level `state_dict` segment (text before the first `.`) for tensors that exist on the Rust
/// inference `VarStore` (trunk + pairformer + MSA + rel_pos + input_embedder + templates, etc.).
pub const BOLTZ2_INFERENCE_TOP_LEVEL_KEYS: &[&str] = &[
    "s_init",
    "z_init_1",
    "z_init_2",
    "s_norm",
    "z_norm",
    "s_recycle",
    "z_recycle",
    "pairformer_module",
    "msa_module",
    "rel_pos",
    "token_bonds",
    "token_bonds_type",
    "contact_conditioning",
    "input_embedder",
    "template_module",
    "diffusion_conditioning",
    "structure_module",
    "distogram_module",
    "bfactor_module",
    "confidence_module",
];

#[inline]
fn top_level_segment(name: &str) -> &str {
    name.split_once('.').map(|(a, _)| a).unwrap_or(name)
}

/// Split checkpoint tensor names into those that belong to the Rust inference VarStore vs
/// everything else (confidence, affinity, EMA copies, etc.).
pub fn partition_safetensors_keys_for_inference(names: &[String]) -> (Vec<String>, Vec<String>) {
    let set: HashSet<&str> = BOLTZ2_INFERENCE_TOP_LEVEL_KEYS.iter().copied().collect();
    let mut infer = Vec::new();
    let mut other = Vec::new();
    for n in names {
        let top = top_level_segment(n);
        if set.contains(top) {
            infer.push(n.clone());
        } else {
            other.push(n.clone());
        }
    }
    infer.sort();
    other.sort();
    (infer, other)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_splits_inference_vs_rest() {
        let names = vec![
            "s_init.weight".to_string(),
            "confidence_head.linear.weight".to_string(),
            "pairformer_module.layers.0.attention.linear_q.weight".to_string(),
            "template_module.z_proj.weight".to_string(),
        ];
        let (inf, oth) = partition_safetensors_keys_for_inference(&names);
        assert_eq!(inf.len(), 3);
        assert_eq!(oth.len(), 1);
        assert!(oth[0].contains("confidence"));
    }
}
