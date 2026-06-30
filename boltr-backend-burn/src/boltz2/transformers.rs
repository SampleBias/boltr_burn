//! Diffusion transformer stack: `AdaLN`, `ConditionedTransitionBlock`,
//! `DiffusionTransformerLayer`, `DiffusionTransformer`, `AtomTransformer`.
//!
//! Reference: `boltz-reference/src/boltz/model/modules/transformersv2.py`

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{sigmoid, silu};
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Tensor};

use crate::attention::AttentionPairBiasV2;
use crate::burn_compat::{layer_norm_no_affine, linear_no_bias, LayerNormWeightOnly};
use crate::tensor_ops::{chunk2, mask_2d_from_1d, repeat_interleave_dim0};

use super::atom_window_keys::windowed_to_keys;

// ---------------------------------------------------------------------------
// AdaLN  (Algorithm 26)
// ---------------------------------------------------------------------------

#[derive(Module, Debug, Clone, Copy)]
pub struct NormNoAffine {
    #[module(skip)]
    dim: usize,
}

impl NormNoAffine {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    pub fn forward<B: Backend>(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        layer_norm_no_affine(x, self.dim)
    }
}

#[derive(Module, Debug)]
pub struct AdaLN<B: Backend> {
    a_norm: NormNoAffine,
    s_norm: LayerNormWeightOnly<B>,
    s_scale: Linear<B>,
    s_bias: Linear<B>,
}

impl<B: Backend> AdaLN<B> {
    pub fn new(device: &Device<B>, dim: usize, dim_single_cond: usize) -> Self {
        let a_norm = NormNoAffine::new(dim);
        let s_norm = LayerNormWeightOnly::new(device, dim_single_cond);
        let s_scale = LinearConfig::new(dim_single_cond, dim).init(device);
        let s_bias = linear_no_bias(device, dim_single_cond, dim);
        Self {
            a_norm,
            s_norm,
            s_scale,
            s_bias,
        }
    }

    pub fn forward(&self, a: Tensor<B, 3>, s: Tensor<B, 3>) -> Tensor<B, 3> {
        let a = self.a_norm.forward(a);
        let s = self.s_norm.forward(s);
        sigmoid(self.s_scale.forward(s.clone())) * a + self.s_bias.forward(s)
    }
}

// ---------------------------------------------------------------------------
// ConditionedTransitionBlock  (Algorithm 25)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct ConditionedTransitionBlock<B: Backend> {
    adaln: AdaLN<B>,
    swish_gate_linear: Linear<B>,
    a_to_b: Linear<B>,
    b_to_a: Linear<B>,
    output_projection_linear: Linear<B>,
}

impl<B: Backend> ConditionedTransitionBlock<B> {
    pub fn new(
        device: &Device<B>,
        dim_single: usize,
        dim_single_cond: usize,
        expansion_factor: Option<usize>,
    ) -> Self {
        let expansion_factor = expansion_factor.unwrap_or(2);
        let dim_inner = dim_single * expansion_factor;

        let adaln = AdaLN::new(device, dim_single, dim_single_cond);
        let swish_gate_linear = linear_no_bias(device, dim_single, dim_inner * 2);
        let a_to_b = linear_no_bias(device, dim_single, dim_inner);
        let b_to_a = linear_no_bias(device, dim_inner, dim_single);
        let output_projection_linear =
            LinearConfig::new(dim_single_cond, dim_single).init(device);

        Self {
            adaln,
            swish_gate_linear,
            a_to_b,
            b_to_a,
            output_projection_linear,
        }
    }

    pub fn forward(&self, a: Tensor<B, 3>, s: Tensor<B, 3>) -> Tensor<B, 3> {
        let a = self.adaln.forward(a, s.clone());
        let gate_out = self.swish_gate_linear.forward(a.clone());
        let (g0, g1) = chunk2(gate_out, 2);
        let swiglu = silu(g1) * g0;
        let b = swiglu * self.a_to_b.forward(a);
        sigmoid(self.output_projection_linear.forward(s)) * self.b_to_a.forward(b)
    }
}

// ---------------------------------------------------------------------------
// Windowed key transform params
// ---------------------------------------------------------------------------

#[derive(Copy, Clone)]
pub struct WindowedKeyParams<'a, B: Backend> {
    pub batch: usize,
    pub n_atoms: usize,
    pub w: usize,
    pub h: usize,
    pub indexing_matrix: &'a Tensor<B, 2>,
}

// ---------------------------------------------------------------------------
// DiffusionTransformerLayer  (Algorithm 23)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct DiffusionTransformerLayer<B: Backend> {
    adaln: AdaLN<B>,
    pair_bias_attn: AttentionPairBiasV2<B>,
    output_projection_linear: Linear<B>,
    transition: ConditionedTransitionBlock<B>,
    #[module(skip)]
    c_s: usize,
}

impl<B: Backend> DiffusionTransformerLayer<B> {
    pub fn new(device: &Device<B>, heads: usize, dim: usize, dim_single_cond: usize) -> Self {
        let adaln = AdaLN::new(device, dim, dim_single_cond);
        let pair_bias_attn = AttentionPairBiasV2::new(device, dim, None, Some(heads), None);
        let output_projection_linear =
            LinearConfig::new(dim_single_cond, dim).init(device);
        let transition =
            ConditionedTransitionBlock::new(device, dim, dim_single_cond, None);
        Self {
            adaln,
            pair_bias_attn,
            output_projection_linear,
            transition,
            c_s: dim,
        }
    }

    pub fn forward(
        &self,
        a: Tensor<B, 3>,
        s: Tensor<B, 3>,
        bias: Option<Tensor<B, 4>>,
        mask: Tensor<B, 2>,
        multiplicity: usize,
        windowed_keys: Option<&WindowedKeyParams<'_, B>>,
    ) -> Tensor<B, 3> {
        let b_val = self.adaln.forward(a.clone(), s.clone());

        let [batch_rows, q_len, _] = b_val.dims();
        let bias_t = match bias {
            Some(b) => b,
            None => Tensor::<B, 4>::zeros([batch_rows, q_len, q_len, 1], &b_val.device()),
        };

        let (k_in, mask_2d, mask_3d) = if let Some(params) = windowed_keys {
            let k_in = windowed_to_keys(
                b_val.clone(),
                params.batch,
                params.n_atoms,
                params.w,
                params.h,
                params.indexing_matrix,
            );
            let mask_m = windowed_to_keys(
                mask.clone().unsqueeze_dim::<3>(2),
                params.batch,
                params.n_atoms,
                params.w,
                params.h,
                params.indexing_matrix,
            );
            let [bk, hk, hd] = mask_m.dims();
            let mask_m = if hd == 1 {
                mask_m.squeeze_dim(2)
            } else {
                mask_m.reshape([bk, hk])
            };
            (k_in, Some(mask_m), None)
        } else {
            let mask_3d = mask_2d_from_1d(mask);
            (b_val.clone(), None, Some(mask_3d))
        };

        let b_val = match (mask_2d, mask_3d) {
            (Some(mask_m), None) => self.pair_bias_attn.forward_key_mask(
                b_val,
                bias_t,
                mask_m,
                k_in,
                Some(multiplicity as i64),
            ),
            (None, Some(mask_3d)) => self.pair_bias_attn.forward(
                b_val,
                bias_t,
                mask_3d,
                k_in,
                Some(multiplicity as i64),
            ),
            _ => unreachable!("windowed and dense masks are mutually exclusive"),
        };

        let b_val = sigmoid(self.output_projection_linear.forward(s.clone())) * b_val;
        let a_out = a + b_val;
        let t = self.transition.forward(a_out.clone(), s);
        a_out + t
    }
}

// ---------------------------------------------------------------------------
// DiffusionTransformer
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct DiffusionTransformer<B: Backend> {
    layers: Vec<DiffusionTransformerLayer<B>>,
    #[module(skip)]
    num_layers: usize,
    #[module(skip)]
    pair_bias_attn: bool,
}

impl<B: Backend> DiffusionTransformer<B> {
    pub fn new(
        device: &Device<B>,
        depth: usize,
        heads: usize,
        dim: usize,
        dim_single_cond: Option<usize>,
        pair_bias_attn: bool,
    ) -> Self {
        let dim_single_cond = dim_single_cond.unwrap_or(dim);
        let mut layers = Vec::with_capacity(depth);
        for _ in 0..depth {
            layers.push(DiffusionTransformerLayer::new(
                device,
                heads,
                dim,
                dim_single_cond,
            ));
        }
        Self {
            layers,
            num_layers: depth,
            pair_bias_attn,
        }
    }

    pub fn forward(
        &self,
        a: Tensor<B, 3>,
        s: Tensor<B, 3>,
        bias: Option<Tensor<B, 4>>,
        mask: Tensor<B, 2>,
        multiplicity: usize,
        windowed_keys: Option<&WindowedKeyParams<'_, B>>,
    ) -> Tensor<B, 3> {
        let mut out = a;

        if self.pair_bias_attn {
            if let Some(bias) = bias {
                let [b, n, m, d] = bias.dims();
                let per_layer = d / self.num_layers;
                let bias_reshaped = bias.reshape([b, n, m, self.num_layers, per_layer]);

                for (i, layer) in self.layers.iter().enumerate() {
                    let bias_l = bias_reshaped.clone().slice([0..b, 0..n, 0..m, i..i + 1, 0..per_layer]);
                    let bias_l = bias_l.squeeze_dim(3);
                    out = layer.forward(
                        out,
                        s.clone(),
                        Some(bias_l),
                        mask.clone(),
                        multiplicity,
                        windowed_keys,
                    );
                }
            } else {
                for layer in &self.layers {
                    out = layer.forward(
                        out,
                        s.clone(),
                        None,
                        mask.clone(),
                        multiplicity,
                        windowed_keys,
                    );
                }
            }
        } else {
            for layer in &self.layers {
                out = layer.forward(
                    out,
                    s.clone(),
                    None,
                    mask.clone(),
                    multiplicity,
                    windowed_keys,
                );
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// AtomTransformer
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct AtomTransformer<B: Backend> {
    #[module(skip)]
    attn_window_queries: usize,
    #[module(skip)]
    attn_window_keys: usize,
    diffusion_transformer: DiffusionTransformer<B>,
}

impl<B: Backend> AtomTransformer<B> {
    pub fn new(
        device: &Device<B>,
        attn_window_queries: usize,
        attn_window_keys: usize,
        depth: usize,
        heads: usize,
        dim: usize,
        dim_single_cond: Option<usize>,
    ) -> Self {
        let diffusion_transformer = DiffusionTransformer::new(
            device,
            depth,
            heads,
            dim,
            dim_single_cond,
            true,
        );
        Self {
            attn_window_queries,
            attn_window_keys,
            diffusion_transformer,
        }
    }

    pub fn forward(
        &self,
        q: Tensor<B, 3>,
        c: Tensor<B, 3>,
        bias: Tensor<B, 5>,
        mask: Tensor<B, 2>,
        multiplicity: usize,
        indexing_matrix: &Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let w = self.attn_window_queries;
        let h_keys = self.attn_window_keys;
        let [b, n, d] = q.dims();
        let nw = n / w;

        let q_w = q.reshape([b * nw, w, d]);
        let c_dim = c.dims()[2];
        let c_w = c.reshape([b * nw, w, c_dim]);

        let bias_exp = if bias.dims()[0] == b {
            bias
        } else {
            repeat_interleave_dim0(bias, multiplicity)
        };
        let [b_bias, k, w, h, heads_dim] = bias_exp.dims();
        let bias_w = bias_exp.reshape([b_bias * k, w, h, heads_dim]);

        let mask_exp = if mask.dims()[0] == b {
            mask
        } else {
            repeat_interleave_dim0(mask, multiplicity)
        };
        let mask_w = mask_exp.reshape([b * nw, w]);

        let windowed_keys = WindowedKeyParams {
            batch: b,
            n_atoms: n,
            w,
            h: h_keys,
            indexing_matrix,
        };

        let out = self.diffusion_transformer.forward(
            q_w,
            c_w,
            Some(bias_w),
            mask_w,
            1,
            Some(&windowed_keys),
        );

        out.reshape([b, nw * w, d])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn adaln_forward_shape() {
        let device = Default::default();
        let dim = 64;
        let dim_cond = 32;
        let adaln = AdaLN::<B>::new(&device, dim, dim_cond);
        let a = Tensor::<B, 3>::random(
            [2, 8, dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let s = Tensor::<B, 3>::random(
            [2, 8, dim_cond],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let out = adaln.forward(a, s);
        assert_eq!(out.dims(), [2, 8, dim]);
    }

    #[test]
    fn conditioned_transition_block_shape() {
        let device = Default::default();
        let dim = 64;
        let ctb = ConditionedTransitionBlock::<B>::new(&device, dim, dim, None);
        let a = Tensor::<B, 3>::random(
            [2, 8, dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let s = Tensor::<B, 3>::random(
            [2, 8, dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let out = ctb.forward(a, s);
        assert_eq!(out.dims(), [2, 8, dim]);
    }

    #[test]
    fn diffusion_transformer_forward_shape() {
        let device = Default::default();
        let dim = 64;
        let depth = 2;
        let heads = 4;
        let b = 2_usize;
        let n = 8_usize;
        let dt = DiffusionTransformer::<B>::new(&device, depth, heads, dim, Some(dim), true);
        let a = Tensor::<B, 3>::random(
            [b, n, dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let s = Tensor::<B, 3>::random(
            [b, n, dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let bias = Tensor::<B, 4>::random(
            [b, n, n, heads * depth],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let mask = Tensor::<B, 2>::ones([b, n], &device);
        let out = dt.forward(a, s, Some(bias), mask, 1, None);
        assert_eq!(out.dims(), [b, n, dim]);
    }
}
