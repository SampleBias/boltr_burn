//! Random rigid augmentation and weighted rigid alignment for diffusion sampling.
//!
//! Reference: `boltz-reference/src/boltz/model/modules/utils.py` (quaternions / rotations),
//! `boltz-reference/src/boltz/model/loss/diffusionv2.py` (`weighted_rigid_align`).

use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Device, Tensor};
use burn::tensor::linalg;

use crate::tensor_ops::{einsum_bmd_bds_bms, weighted_cov_bni_bnj_bij};

/// Random rotation matrices `(multiplicity, 3, 3)` and translation `(multiplicity, 1, 3)`.
pub fn compute_random_augmentation<B: Backend>(
    multiplicity: usize,
    device: &Device<B>,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let quat = random_quaternions::<B>(multiplicity, device);
    let r = quaternion_to_matrix(quat);
    let random_tr = Tensor::<B, 3>::random(
        [multiplicity, 1, 3],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );
    (r, random_tr)
}

fn random_quaternions<B: Backend>(n: usize, device: &Device<B>) -> Tensor<B, 2> {
    let o = Tensor::<B, 2>::random(
        [n, 4],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );
    let s = (o.clone() * o.clone()).sum_dim(1);
    let sgn = o.clone().slice([0..n, 0..1]).sign();
    let sqrt_s = s.sqrt();
    o * (sgn / sqrt_s)
}

fn quaternion_to_matrix<B: Backend>(quaternions: Tensor<B, 2>) -> Tensor<B, 3> {
    let [n, _] = quaternions.dims();
    let device = quaternions.device();
    let data = quaternions.into_data();
    let slice = data.as_slice::<f32>().expect("quaternion f32");
    let mut out = vec![0.0f32; n * 9];
    for i in 0..n {
        let r = slice[i * 4];
        let xi = slice[i * 4 + 1];
        let yj = slice[i * 4 + 2];
        let zk = slice[i * 4 + 3];
        out[i * 9] = 1.0 - 2.0 * (yj * yj + zk * zk);
        out[i * 9 + 1] = 2.0 * (xi * yj - zk * r);
        out[i * 9 + 2] = 2.0 * (xi * zk + yj * r);
        out[i * 9 + 3] = 2.0 * (xi * yj + zk * r);
        out[i * 9 + 4] = 1.0 - 2.0 * (xi * xi + zk * zk);
        out[i * 9 + 5] = 2.0 * (yj * zk - xi * r);
        out[i * 9 + 6] = 2.0 * (xi * zk - yj * r);
        out[i * 9 + 7] = 2.0 * (yj * zk + xi * r);
        out[i * 9 + 8] = 1.0 - 2.0 * (xi * xi + yj * yj);
    }
    Tensor::<B, 3>::from_data(
        burn::tensor::TensorData::new(out, [n, 3, 3]),
        &device,
    )
}

/// Jacobi SVD for a single 3×3 matrix. Returns `(U, Vh)`.
#[allow(clippy::needless_range_loop)]
fn svd3x3(a: [[f64; 3]; 3]) -> ([[f64; 3]; 3], [[f64; 3]; 3]) {
    let mut u = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut b = a;

    for _ in 0..50 {
        for p in 0..2 {
            for q in (p + 1)..3 {
                let apq = b[p][q];
                if apq.abs() < 1e-15 {
                    continue;
                }
                let app = b[p][p];
                let aqq = b[q][q];
                let tau = (aqq - app) / (2.0 * apq);
                let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                for k in 0..3 {
                    let bp = b[p][k];
                    let bq = b[q][k];
                    b[p][k] = c * bp - s * bq;
                    b[q][k] = s * bp + c * bq;
                }
                for k in 0..3 {
                    let up = u[k][p];
                    let uq = u[k][q];
                    u[k][p] = c * up - s * uq;
                    u[k][q] = s * up + c * uq;
                }
                for k in 0..3 {
                    let vp = v[p][k];
                    let vq = v[q][k];
                    v[p][k] = c * vp - s * vq;
                    v[q][k] = s * vp + c * vq;
                }
            }
        }
    }

    let mut vh = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            vh[i][j] = v[j][i];
        }
    }
    (u, vh)
}

fn batch_svd3x3<B: Backend>(cov: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let device = cov.device();
    let [batch, _, _] = cov.dims();
    let data = cov.to_data();
    let slice = data.as_slice::<f32>().expect("cov f32");
    let mut u_flat = vec![0.0f32; batch * 9];
    let mut vh_flat = vec![0.0f32; batch * 9];

    for b in 0..batch {
        let mut a = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] = slice[b * 9 + i * 3 + j] as f64;
            }
        }
        let (u, vh) = svd3x3(a);
        for i in 0..3 {
            for j in 0..3 {
                u_flat[b * 9 + i * 3 + j] = u[i][j] as f32;
                vh_flat[b * 9 + i * 3 + j] = vh[i][j] as f32;
            }
        }
    }

    let u = Tensor::<B, 3>::from_data(
        burn::tensor::TensorData::new(u_flat, [batch, 3, 3]),
        &device,
    );
    let vh = Tensor::<B, 3>::from_data(
        burn::tensor::TensorData::new(vh_flat, [batch, 3, 3]),
        &device,
    );
    (u, vh)
}

/// Algorithm 28 (`weighted_rigid_align`): align `true_coords` to `pred_coords` with weights.
pub fn weighted_rigid_align<B: Backend>(
    true_coords: Tensor<B, 3>,
    pred_coords: Tensor<B, 3>,
    weights: Tensor<B, 2>,
    mask: Tensor<B, 2, Bool>,
) -> Tensor<B, 3> {
    let device = true_coords.device();
    let weights = mask.float() * weights;
    let weights = weights.unsqueeze_dim::<3>(2);

    let wsum = weights.clone().sum_dim(1);
    let true_centroid = (true_coords.clone() * weights.clone()).sum_dim(1) / wsum.clone();
    let pred_centroid = (pred_coords.clone() * weights.clone()).sum_dim(1) / wsum;

    let true_centered = true_coords - true_centroid.clone().unsqueeze_dim::<3>(1);
    let pred_centered = pred_coords - pred_centroid.clone().unsqueeze_dim::<3>(1);

    let cov_matrix = weighted_cov_bni_bnj_bij(
        pred_centered.clone(),
        true_centered.clone(),
        weights.clone(),
    );

    let (u, vh) = batch_svd3x3(cov_matrix);
    let rot_matrix = u.clone().matmul(vh.clone());

    let dim = 3_usize;
    let batch = rot_matrix.dims()[0];
    let eye = Tensor::<B, 2>::eye(dim, &device);
    let f = if batch > 1 {
        eye.clone().unsqueeze_dim::<3>(0).repeat(&[batch, 1, 1])
    } else {
        eye.unsqueeze_dim::<3>(0)
    };

    let det = linalg::det::<B, 3, 2, 1>(rot_matrix.clone());
    let corner = Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [3, 3]),
        &device,
    );
    let corner = if batch > 1 {
        corner.unsqueeze_dim::<3>(0).repeat(&[batch, 1, 1])
    } else {
        corner.unsqueeze_dim::<3>(0)
    };
    let f = f + (det.unsqueeze_dim::<3>(1).unsqueeze_dim::<3>(2) - 1.0) * corner;

    let rot_matrix = u.matmul(f).matmul(vh);
    let aligned = einsum_bmd_bds_bms(true_centered, rot_matrix.transpose())
        + pred_centroid.clone().unsqueeze_dim::<3>(1);
    aligned.detach()
}
