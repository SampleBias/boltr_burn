//! Tensor ops mirroring PyTorch / tch paths used in Boltz2.

use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

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

pub fn repeat_interleave_dim0<B: Backend, const D: usize>(
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

/// `torch.einsum("bijd,bwki,bwlj->bwkld", z, q, keys)`.
pub fn einsum_bijd_bwki_bwlj_bwkld<B: Backend>(
    z: Tensor<B, 4>,
    q: Tensor<B, 4>,
    keys: Tensor<B, 4>,
) -> Tensor<B, 5> {
    let z6 = z.unsqueeze_dim::<6>(1).unsqueeze_dim::<6>(2);
    let q6 = q.unsqueeze_dim::<6>(5).unsqueeze_dim::<6>(4);
    let t = (z6 * q6).sum_dim(3);
    let keys6 = keys.unsqueeze_dim::<6>(3).unsqueeze_dim::<6>(5);
    let t6 = t.unsqueeze_dim::<6>(3);
    (t6 * keys6).sum_dim(4).squeeze_dim(4)
}
