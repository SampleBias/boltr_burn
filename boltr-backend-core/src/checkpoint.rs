//! Backend-agnostic safetensors helpers (no ML framework dependency).

use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result};
use safetensors::SafeTensors;

/// List tensor names in a safetensors file (for debugging / key alignment).
pub fn list_safetensor_names(path: &Path) -> Result<Vec<String>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let st = SafeTensors::deserialize(&bytes).context("parse safetensors")?;
    Ok(st.names().into_iter().map(String::from).collect())
}

/// Keys present in the file but not in `expected` (unused by this graph, or naming mismatch).
pub fn safetensor_names_not_in_expected(path: &Path, expected: &[String]) -> Result<Vec<String>> {
    let file_keys: HashSet<String> = list_safetensor_names(path)?.into_iter().collect();
    let expected_keys: HashSet<String> = expected.iter().cloned().collect();
    let mut extra: Vec<String> = file_keys.difference(&expected_keys).cloned().collect();
    extra.sort();
    Ok(extra)
}

/// Parameter names that have **no** tensor in the safetensors file (pre-load check).
pub fn expected_keys_missing_in_safetensors(path: &Path, expected: &[String]) -> Result<Vec<String>> {
    let file_keys: HashSet<String> = list_safetensor_names(path)?.into_iter().collect();
    let mut missing: Vec<String> = expected
        .iter()
        .filter(|k| !file_keys.contains(*k))
        .cloned()
        .collect();
    missing.sort();
    Ok(missing)
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::Dtype;
    use std::collections::HashMap;

    #[test]
    fn missing_and_extra_keys() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.safetensors");

        let mut tensors = HashMap::new();
        let a_data: Vec<u8> = vec![0, 0, 128, 63];
        let b_data: Vec<u8> = vec![0, 0, 128, 63];
        tensors.insert(
            "a.weight".to_string(),
            safetensors::tensor::TensorView::new(Dtype::F32, vec![1], &a_data).unwrap(),
        );
        tensors.insert(
            "b.weight".to_string(),
            safetensors::tensor::TensorView::new(Dtype::F32, vec![1], &b_data).unwrap(),
        );
        std::fs::write(
            &path,
            safetensors::serialize(tensors, &None).unwrap(),
        )
        .unwrap();

        let expected = vec!["a.weight".to_string(), "c.weight".to_string()];
        let missing = expected_keys_missing_in_safetensors(&path, &expected).unwrap();
        assert_eq!(missing, vec!["c.weight"]);

        let extra = safetensor_names_not_in_expected(&path, &expected).unwrap();
        assert_eq!(extra, vec!["b.weight"]);
    }
}
