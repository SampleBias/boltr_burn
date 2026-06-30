//! Tensor ops mirroring PyTorch / tch paths used in Boltz2.

use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

fn repeat_interleave_impl<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    multiplicity: usize,
) -> Tensor<B, D> {
    if multiplicity <= 1 {
        return x;
    }
    let mut dims = x.dims();
    let batch = dims[0];
    dims[0] = batch * multiplicity;
    let inner: usize = x.dims()[1..].iter().product();
    x.reshape([batch, inner])
        .unsqueeze_dim::<2>(1)
        .repeat(&[1, multiplicity, 1])
        .reshape(dims)
}

fn repeat_interleave_impl_int<B: Backend, const D: usize>(
    x: Tensor<B, D, Int>,
    multiplicity: usize,
) -> Tensor<B, D, Int> {
    if multiplicity <= 1 {
        return x;
    }
    let mut dims = x.dims();
    let batch = dims[0];
    dims[0] = batch * multiplicity;
    let inner: usize = x.dims()[1..].iter().product();
    x.reshape([batch, inner])
        .unsqueeze_dim::<2>(1)
        .repeat(&[1, multiplicity, 1])
        .reshape(dims)
}

/// `torch.einsum("bikd,bjkd->bijd", a, b)`.
pub fn einsum_bikd_bjkd_bijd<B: Backend>(
    a: Tensor<B, 4>,
    b: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let a = a.unsqueeze_dim::<5>(2);
    let b = b.unsqueeze_dim::<5>(1);
    (a * b).sum_dim(3).squeeze_dim(3)
}

/// `torch.einsum("bkid,bkjd->bijd", a, b)`.
pub fn einsum_bkid_bkjd_bijd<B: Backend>(
    a: Tensor<B, 4>,
    b: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let a = a.swap_dims(1, 2);
    let b = b.swap_dims(1, 2);
    einsum_bikd_bjkd_bijd(a, b)
}

/// `torch.einsum("bihd,bjhd->bhij", q, k)`.
pub fn einsum_bihd_bjhd_bhij<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let q = q.swap_dims(1, 2);
    let k = k.swap_dims(1, 2);
    q.matmul(k.swap_dims(2, 3))
}

/// `torch.einsum("bhij,bjhd->bihd", attn, v)`.
pub fn einsum_bhij_bjhd_bihd<B: Backend>(
    attn: Tensor<B, 4>,
    v: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let v = v.swap_dims(1, 2);
    attn.matmul(v).swap_dims(1, 2)
}

/// Integer one-hot: `[B,N,M]` -> float `[B,N,M,C]`.
pub fn one_hot_int<B: Backend>(
    indices: Tensor<B, 3, Int>,
    num_classes: usize,
) -> Tensor<B, 4> {
    let [b, n, m] = indices.dims();
    let device = indices.device();
    let mut planes = Vec::with_capacity(num_classes);
    for c in 0..num_classes {
        let c_val = Tensor::<B, 3, Int>::full([b, n, m], c as i64, &device);
        planes.push(indices.clone().equal(c_val).float());
    }
    Tensor::stack(planes, 3)
}

/// `torch.where(cond, a, b)` for int tensors.
pub fn where_int<B: Backend>(
    cond: Tensor<B, 3, Bool>,
    a: Tensor<B, 3, Int>,
    b: Tensor<B, 3, Int>,
) -> Tensor<B, 3, Int> {
    let cf = cond.float();
    let one = Tensor::<B, 3>::ones(a.dims(), &a.device());
    (a.float() * cf.clone() + b.float() * (one - cf)).int()
}

/// Split into two equal chunks along `dim`.
pub fn chunk2<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    dim: usize,
) -> (Tensor<B, D>, Tensor<B, D>) {
    let parts = x.chunk(2, dim);
    (parts[0].clone(), parts[1].clone())
}

/// Broadcast `mask [B,N]` to `[B,N,N]` for symmetric token attention.
pub fn mask_2d_from_1d<B: Backend>(mask: Tensor<B, 2>) -> Tensor<B, 3> {
    let [b, n] = mask.dims();
    mask.unsqueeze_dim::<3>(1)
        .repeat(&[1, n, 1])
        .reshape([b, n, n])
}

/// `torch.einsum("bhij,bhsjd->bhsid", w, v)`.
pub fn einsum_bhij_bhsjd_bhsid<B: Backend>(
    w: Tensor<B, 4>,
    v: Tensor<B, 5>,
) -> Tensor<B, 5> {
    let [batch, heads, i_len, _] = w.dims();
    let [_, _, s_len, _, d] = v.dims();
    let w = w.unsqueeze_dim::<5>(2).unsqueeze_dim::<6>(5);
    let v = v.unsqueeze_dim::<6>(3);
    (w * v).sum_dim(4).reshape([batch, heads, s_len, i_len, d])
}

/// `torch.einsum("bsic,bsjd->bijcd", a, b)`.
pub fn einsum_bsic_bsjd_bijcd<B: Backend>(
    a: Tensor<B, 4>,
    b: Tensor<B, 4>,
) -> Tensor<B, 5> {
    let [batch, _, i_len, c] = a.dims();
    let [_, _, j_len, d] = b.dims();
    let a = a.unsqueeze_dim::<5>(3).unsqueeze_dim::<6>(5);
    let b = b.unsqueeze_dim::<5>(2).unsqueeze_dim::<6>(4);
    (a * b).sum_dim(1).squeeze_dim::<5>(1).reshape([batch, i_len, j_len, c, d])
}

/// Pairwise L2 distance: `[B, N, D]` → `[B, N, N]`.
pub fn cdist_euclidean<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let sq = x.clone().powf_scalar(2.0).sum_dim(2).squeeze_dim::<2>(2);
    let inner = x.clone().matmul(x.clone().swap_dims(1, 2));
    (sq.clone().unsqueeze_dim::<3>(2) + sq.unsqueeze_dim::<3>(1) - inner.mul_scalar(2.0))
        .clamp_min(0.0)
        .sqrt()
}

/// Ascending linspace (inclusive endpoints), matching `torch.linspace`.
pub fn linspace(start: f64, end: f64, steps: usize) -> Vec<f64> {
    if steps == 0 {
        return vec![];
    }
    if steps == 1 {
        return vec![start];
    }
    (0..steps)
        .map(|i| start + (end - start) * i as f64 / (steps - 1) as f64)
        .collect()
}

/// Bin index from distances and ascending boundaries (count `d > boundary`).
pub fn dist_to_bin_indices<B: Backend>(
    d: Tensor<B, 3>,
    boundaries: &[f64],
) -> Tensor<B, 3, Int> {
    let device = d.device();
    let dims = d.dims();
    let mut bin_idx = Tensor::<B, 3, Int>::zeros(dims, &device);
    for &boundary in boundaries {
        let bnd = Tensor::<B, 3>::full(dims, boundary, &device);
        bin_idx = bin_idx + d.clone().greater(bnd).int();
    }
    bin_idx
}

pub fn repeat_interleave_dim0<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    multiplicity: usize,
) -> Tensor<B, D> {
    repeat_interleave_impl(x, multiplicity)
}

pub fn repeat_interleave_dim0_int<B: Backend, const D: usize>(
    x: Tensor<B, D, Int>,
    multiplicity: usize,
) -> Tensor<B, D, Int> {
    repeat_interleave_impl_int(x, multiplicity)
}

/// Integer one-hot: `[B,N]` -> float `[B,N,C]`.
pub fn one_hot_2d<B: Backend>(
    indices: Tensor<B, 2, Int>,
    num_classes: usize,
) -> Tensor<B, 3> {
    let [b, n] = indices.dims();
    let device = indices.device();
    let mut planes = Vec::with_capacity(num_classes);
    for c in 0..num_classes {
        let c_val = Tensor::<B, 2, Int>::full([b, n], c as i64, &device);
        planes.push(indices.clone().equal(c_val).float());
    }
    Tensor::stack(planes, 2)
}

/// `torch.einsum("bjid,jk->bkid", a, indexing_matrix)`.
pub fn einsum_bjid_jk_bkid<B: Backend>(
    a: Tensor<B, 4>,
    indexing_matrix: Tensor<B, 2>,
) -> Tensor<B, 4> {
    let [_, j_len, i_len, d] = a.dims();
    let im = indexing_matrix
        .clone()
        .reshape([j_len, indexing_matrix.dims()[1], 1, 1]);
    let batch = a.dims()[0];
    let a5 = a.unsqueeze_dim::<5>(2);
    (a5 * im.unsqueeze_dims(&[0, 3, 4])).sum_dim(1).reshape([batch, indexing_matrix.dims()[1], i_len, d])
}

/// `torch.einsum("bmd,bds->bms", a, r)`.
pub fn einsum_bmd_bds_bms<B: Backend>(
    a: Tensor<B, 3>,
    r: Tensor<B, 3>,
) -> Tensor<B, 3> {
    a.matmul(r)
}

/// `torch.einsum("bni,bnj->bij", a, b)` — sum over the atom/token dimension.
pub fn einsum_bni_bnj_bij<B: Backend>(
    a: Tensor<B, 3>,
    b: Tensor<B, 3>,
) -> Tensor<B, 3> {
    a.swap_dims(1, 2).matmul(b)
}

/// Weighted covariance `einsum("bni,bnj->bij", weights * pred, true)`.
pub fn weighted_cov_bni_bnj_bij<B: Backend>(
    pred: Tensor<B, 3>,
    true_coords: Tensor<B, 3>,
    weights: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let wp = pred * weights.clone();
    wp.swap_dims(1, 2).matmul(true_coords)
}

/// `torch.einsum("bijd,bwki,bwlj->bwkld", z, q, keys)`.
pub fn einsum_bijd_bwki_bwlj_bwkld<B: Backend>(
    z: Tensor<B, 4>,
    q: Tensor<B, 4>,
    keys: Tensor<B, 4>,
) -> Tensor<B, 5> {
    let [b, n_i, n_j, d] = z.dims();
    let [_, k_win, w_q, n_q] = q.dims();
    let [_, k2, h_keys, _] = keys.dims();
    debug_assert_eq!(n_i, n_q, "einsum_bijd_bwki_bwlj_bwkld: token dim mismatch");
    debug_assert_eq!(k_win, k2, "einsum_bijd_bwki_bwlj_bwkld: window dim mismatch");

    // Step 1: t[b,k,w,j,d] = sum_i z[b,i,j,d] * q[b,k,w,i]
    let q_flat = q.reshape([b, k_win * w_q, n_i]);
    let z_flat = z.reshape([b, n_i, n_j * d]);
    let t = q_flat.matmul(z_flat).reshape([b, k_win, w_q, n_j, d]);

    // Step 2: out[b,k,l,w,d] = sum_j t[b,k,w,j,d] * keys[b,k,l,j]
    let t_perm = t.swap_dims(2, 3);
    let t_mat = t_perm.reshape([b * k_win, n_j, w_q * d]);
    let keys_mat = keys.reshape([b * k_win, h_keys, n_j]);
    keys_mat.matmul(t_mat).reshape([b, k_win, h_keys, w_q, d])
}
