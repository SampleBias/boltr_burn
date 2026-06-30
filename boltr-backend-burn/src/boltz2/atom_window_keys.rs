//! Windowed key indexing shared by [`super::encoders::AtomEncoder`] and [`super::transformers::AtomTransformer`].
//!
//! Reference: `boltz-reference/src/boltz/model/modules/encodersv2.py` (`single_to_keys`, `get_indexing_matrix`).

use burn::tensor::backend::Backend;
use burn::tensor::{Device, Int, Tensor};

use crate::tensor_ops::einsum_bjid_jk_bkid;

/// Build the static indexing matrix for windowed attention keys.
/// Equivalent to Python `get_indexing_matrix(K, W, H, device)`.
pub fn get_indexing_matrix<B: Backend>(
    k: usize,
    w: usize,
    h: usize,
    device: &Device<B>,
) -> Tensor<B, 2> {
    assert!(w.is_multiple_of(2), "W must be even");
    let half_w = w / 2;
    assert!(h.is_multiple_of(half_w), "H must be divisible by W/2");
    let h_ratio = h / half_w;
    assert!(h_ratio.is_multiple_of(2), "h ratio must be even");

    let n = 2 * k;
    let arange = Tensor::<B, 1, Int>::arange(0..n as i64, device).reshape([n, 1]);
    let diff = arange.clone() - arange.transpose();
    let index = (diff + h_ratio as i64 / 2)
        .clamp(0, (h_ratio + 1) as i64)
        .reshape([k, 2, 2 * k])
        .slice([0..k, 0..1, 0..2 * k])
        .squeeze_dim(1);

    let num_classes = h_ratio + 2;
    let [rows, cols] = index.dims();
    let mut planes: Vec<Tensor<B, 3>> = Vec::with_capacity(num_classes);
    for class in 0..num_classes {
        let c_val = Tensor::<B, 2, Int>::full([rows, cols], class as i64, device);
        planes.push(index.clone().equal(c_val).float().unsqueeze_dim::<3>(2));
    }
    let onehot: Tensor<B, 3> = Tensor::cat(planes, 2);
    let onehot = onehot.slice([0..rows, 0..cols, 1..num_classes - 1]);
    onehot.swap_dims(0, 1)
        .reshape([2 * k, h_ratio * k])
}

/// Map single representation from query windows to key windows.
/// `single [B, N, D]` → `[B, K, H, D]`.
pub fn single_to_keys<B: Backend>(
    single: Tensor<B, 3>,
    indexing_matrix: &Tensor<B, 2>,
    w: usize,
    h: usize,
) -> Tensor<B, 4> {
    let [b, n, d] = single.dims();
    let k = n / w;
    let single_r = single.reshape([b, 2 * k, w / 2, d]);
    let out = einsum_bjid_jk_bkid(single_r, indexing_matrix.clone());
    out.reshape([b, k, h, d])
}

/// Boltz `AtomTransformer.to_keys_new`: map `[B*NW, W, D]` → `[B*NW, H, D]`.
pub fn windowed_to_keys<B: Backend>(
    x: Tensor<B, 3>,
    batch: usize,
    n_atoms: usize,
    w: usize,
    h: usize,
    indexing_matrix: &Tensor<B, 2>,
) -> Tensor<B, 3> {
    let [rows, width, d] = x.dims();
    let nw = n_atoms / w;
    assert_eq!(rows, batch * nw, "windowed_to_keys: batch*NW mismatch");
    assert_eq!(width, w, "windowed_to_keys: W mismatch");
    let x_flat = x.reshape([batch, n_atoms, d]);
    let sk = single_to_keys(x_flat, indexing_matrix, w, h);
    sk.reshape([batch * nw, h, d])
}
