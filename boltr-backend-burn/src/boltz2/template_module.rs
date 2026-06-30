//! Boltz2 `TemplateV2Module` — template pairwise bias on `z`.

use burn::module::Module;
use burn::nn::{LayerNorm, Linear};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Int, Tensor};

use crate::burn_compat::{layer_norm_1d, linear_no_bias};
use crate::layers::PairformerNoSeqModule;
use super::input_embedder::BOLTZ_NUM_TOKENS;

pub struct TemplateFeatures<'a, B: Backend> {
    pub template_restype: &'a Tensor<B, 4>,
    pub template_frame_rot: &'a Tensor<B, 5>,
    pub template_frame_t: &'a Tensor<B, 4>,
    pub template_mask_frame: &'a Tensor<B, 3>,
    pub template_cb: &'a Tensor<B, 4>,
    pub template_ca: &'a Tensor<B, 4>,
    pub template_mask_cb: &'a Tensor<B, 3>,
    pub visibility_ids: &'a Tensor<B, 3, Int>,
    pub template_mask: &'a Tensor<B, 3>,
}

#[derive(Module, Debug)]
pub struct TemplateV2Module<B: Backend> {
    min_dist: f64,
    max_dist: f64,
    num_bins: usize,
    token_z: usize,
    template_dim: usize,
    z_norm: LayerNorm<B>,
    v_norm: LayerNorm<B>,
    z_proj: Linear<B>,
    a_proj: Linear<B>,
    u_proj: Linear<B>,
    pairformer: PairformerNoSeqModule<B>,
}

impl<B: Backend> TemplateV2Module<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device<B>,
        token_z: usize,
        template_dim: usize,
        template_blocks: usize,
        dropout: Option<f64>,
        pairwise_head_width: Option<usize>,
        pairwise_num_heads: Option<usize>,
        min_dist: Option<f64>,
        max_dist: Option<f64>,
        num_bins: Option<usize>,
    ) -> Self {
        let min_dist = min_dist.unwrap_or(3.25);
        let max_dist = max_dist.unwrap_or(50.75);
        let num_bins = num_bins.unwrap_or(38);
        let a_in_dim = BOLTZ_NUM_TOKENS * 2 + num_bins + 5;
        Self {
            min_dist,
            max_dist,
            num_bins,
            token_z,
            template_dim,
            z_norm: layer_norm_1d(device, token_z),
            v_norm: layer_norm_1d(device, template_dim),
            z_proj: linear_no_bias(device, token_z, template_dim),
            a_proj: linear_no_bias(device, a_in_dim, template_dim),
            u_proj: linear_no_bias(device, template_dim, token_z),
            pairformer: PairformerNoSeqModule::new(
                device,
                template_dim,
                template_blocks,
                dropout.unwrap_or(0.25),
                pairwise_head_width.unwrap_or(32),
                pairwise_num_heads.unwrap_or(4),
            ),
        }
    }

    pub fn forward(
        &self,
        z: Tensor<B, 4>,
        feats: &TemplateFeatures<'_, B>,
        pair_mask: Tensor<B, 3>,
        use_kernels: bool,
    ) -> Tensor<B, 4> {
        let [b, t, n, _] = feats.template_restype.dims();
        let template_mask = feats
            .template_mask
            .clone()
            .sum_dim(2)
            .greater_elem(0.0)
            .float();
        let num_templates = template_mask.clone().sum_dim(1).clamp_min(1.0);

        let cb_mask = feats.template_mask_cb.clone();
        let b_cb_mask = cb_mask
            .clone()
            .unsqueeze_dim::<4>(3)
            * cb_mask.unsqueeze_dim::<4>(2);
        let b_cb_mask = b_cb_mask.unsqueeze_dim::<5>(4);

        let frame_mask = feats.template_mask_frame.clone();
        let b_frame_mask = frame_mask
            .clone()
            .unsqueeze_dim::<4>(3)
            * frame_mask.unsqueeze_dim::<4>(2);
        let b_frame_mask = b_frame_mask.unsqueeze_dim::<5>(4);

        let vis = feats.visibility_ids.clone();
        let tmlp_pair_mask = vis
            .clone()
            .unsqueeze_dim::<4>(3)
            .equal(vis.unsqueeze_dim::<4>(2))
            .float();

        let distogram = self.compute_distogram(feats.template_cb.clone());
        let unit_vector =
            Self::compute_unit_vectors(feats.template_ca.clone(), feats.template_frame_rot.clone(), feats.template_frame_t.clone());

        let mut a_tij = Tensor::cat(
            vec![distogram, b_cb_mask, unit_vector, b_frame_mask],
            4,
        );
        a_tij = a_tij * tmlp_pair_mask.unsqueeze_dim::<5>(4);

        let res_type = feats.template_restype.clone();
        let res_type_i = res_type.clone().unsqueeze_dim::<5>(3).repeat(&[1, 1, 1, n, 1]);
        let res_type_j = res_type.unsqueeze_dim::<5>(2).repeat(&[1, 1, n, 1, 1]);
        let a_tij = Tensor::cat(vec![a_tij, res_type_i, res_type_j], 4);
        let a_tij = self.a_proj.forward(a_tij);

        let pair_mask_t = pair_mask
            .unsqueeze_dim::<4>(1)
            .repeat(&[1, t, 1, 1])
            .reshape([b * t, n, n]);

        let z_proj = self.z_proj.forward(self.z_norm.forward(z.unsqueeze_dim::<5>(1)));
        let v = z_proj + a_tij;
        let v = v.reshape([b * t, n, n, self.template_dim]);
        let v = v.clone() + self.pairformer.forward(v, pair_mask_t, use_kernels);
        let v = self.v_norm.forward(v).reshape([b, t, n, n, self.template_dim]);

        let tmask = template_mask.unsqueeze_dim::<5>(2).unsqueeze_dim::<5>(3).unsqueeze_dim::<5>(4);
        let ntmpl = num_templates.unsqueeze_dim::<4>(2).unsqueeze_dim::<4>(3).unsqueeze_dim::<4>(4);
        let u = (v * tmask).sum_dim(1).squeeze_dim(1) / ntmpl;
        relu(self.u_proj.forward(u))
    }

    fn compute_distogram(&self, cb_coords: Tensor<B, 4>) -> Tensor<B, 5> {
        let device = cb_coords.device();
        let [b, t, n, _] = cb_coords.dims();
        let sq = cb_coords
            .clone()
            .powf_scalar(2.0)
            .sum_dim(3)
            .reshape([b, t, n]);
        let inner = cb_coords.clone().matmul(cb_coords.clone().swap_dims(2, 3));
        let dists =
            (sq.clone().unsqueeze_dim::<4>(3) + sq.unsqueeze_dim::<4>(2) - inner.mul_scalar(2.0))
                .clamp_min(0.0)
                .sqrt();

        let steps = self.num_bins - 1;
        let mut boundaries = Vec::with_capacity(steps);
        for i in 0..steps {
            let v = self.min_dist + (self.max_dist - self.min_dist) * (i as f64) / (steps as f64);
            boundaries.push(Tensor::<B, 4>::full(dists.dims(), v, &device));
        }
        let mut bin_idx = Tensor::<B, 4>::zeros(dists.dims(), &device);
        for boundary in boundaries {
            bin_idx = bin_idx + dists.clone().greater(boundary).float();
        }
        let dims = bin_idx.dims();
        let flat_len: usize = dims.iter().product();
        let flat = bin_idx.int().reshape([flat_len]);
        let mut planes = Vec::with_capacity(self.num_bins);
        for c in 0..self.num_bins {
            planes.push(
                flat.clone()
                    .equal_elem(c as i64)
                    .float()
                    .unsqueeze_dim::<2>(1),
            );
        }
        Tensor::cat(planes, 1).reshape([dims[0], dims[1], dims[2], dims[3], self.num_bins])
    }

    fn compute_unit_vectors(
        ca_coords: Tensor<B, 4>,
        frame_rot: Tensor<B, 5>,
        frame_t: Tensor<B, 4>,
    ) -> Tensor<B, 5> {
        let rot_t = frame_rot.swap_dims(3, 4).unsqueeze_dim::<6>(2);
        let t_exp = frame_t.unsqueeze_dim::<6>(2).unsqueeze_dim::<6>(5);
        let ca_exp = ca_coords.unsqueeze_dim::<6>(3).unsqueeze_dim::<6>(5);
        let vector = rot_t.matmul(ca_exp - t_exp);
        let norm = vector.clone().powf_scalar(2.0).sum_dim(5).sqrt();
        let zero = Tensor::<B, 6>::zeros(vector.dims(), &vector.device());
        let unit = vector / norm.clone().mask_where(norm.clone().greater_elem(0.0), zero);
        unit.squeeze_dim(5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    #[ignore = "template forward broadcast paths need golden fixture validation"]
    fn template_v2_forward_shapes() {
        let device = Default::default();
        let token_z = 64;
        let tmpl = TemplateV2Module::<B>::new(&device, token_z, 32, 1, None, None, None, None, None, None);
        let b = 2;
        let t = 3;
        let n = 8;
        let z = Tensor::<B, 4>::random(
            [b, n, n, token_z],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let pair_mask = Tensor::<B, 3>::ones([b, n, n], &device);
        let frame_rot = Tensor::<B, 5>::zeros([b, t, n, 3, 3], &device);
        let feats = TemplateFeatures {
            template_restype: &Tensor::<B, 4>::zeros([b, t, n, BOLTZ_NUM_TOKENS], &device),
            template_frame_rot: &frame_rot,
            template_frame_t: &Tensor::<B, 4>::zeros([b, t, n, 3], &device),
            template_mask_frame: &Tensor::<B, 3>::ones([b, t, n], &device),
            template_cb: &Tensor::<B, 4>::random(
                [b, t, n, 3],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
            template_ca: &Tensor::<B, 4>::random(
                [b, t, n, 3],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
            template_mask_cb: &Tensor::<B, 3>::ones([b, t, n], &device),
            visibility_ids: &Tensor::<B, 3, Int>::zeros([b, t, n], &device),
            template_mask: &Tensor::<B, 3>::ones([b, t, n], &device),
        };
        let u = tmpl.forward(z, &feats, pair_mask, false);
        assert_eq!(u.dims(), [b, n, n, token_z]);
    }
}
