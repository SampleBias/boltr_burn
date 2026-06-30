//! Pairformer layer and module stack.

use burn::module::Module;
use burn::nn::LayerNorm;
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Tensor};

use crate::attention::pair_bias::AttentionPairBiasV2;
use crate::burn_compat::layer_norm_1d;
use crate::layers::transition::Transition;
use crate::layers::triangular_attention::TriangleAttention;
use crate::layers::triangular_mult::{TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing};

#[derive(Module, Debug)]
pub struct PairformerLayer<B: Backend> {
    pre_norm_s: LayerNorm<B>,
    attention: AttentionPairBiasV2<B>,
    transition_s: Transition<B>,
    s_post_norm: Option<LayerNorm<B>>,
    tri_mul_out: TriangleMultiplicationOutgoing<B>,
    tri_mul_in: TriangleMultiplicationIncoming<B>,
    tri_att_start: TriangleAttention<B>,
    tri_att_end: TriangleAttention<B>,
    transition_z: Transition<B>,
    token_z: usize,
    dropout: f64,
}

impl<B: Backend> PairformerLayer<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device<B>,
        token_s: usize,
        token_z: usize,
        num_heads: usize,
        dropout: f64,
        pairwise_head_width: usize,
        pairwise_num_heads: usize,
        post_layer_norm: bool,
    ) -> Self {
        Self {
            pre_norm_s: layer_norm_1d(device, token_s),
            attention: AttentionPairBiasV2::new(
                device,
                token_s,
                Some(token_z),
                Some(num_heads),
                Some(1e6),
            ),
            transition_s: Transition::new(device, token_s, Some(token_s * 4), None),
            s_post_norm: post_layer_norm.then(|| layer_norm_1d(device, token_s)),
            tri_mul_out: TriangleMultiplicationOutgoing::new(device, token_z),
            tri_mul_in: TriangleMultiplicationIncoming::new(device, token_z),
            tri_att_start: TriangleAttention::new(
                device,
                token_z,
                pairwise_head_width,
                pairwise_num_heads,
                true,
                1e9,
            ),
            tri_att_end: TriangleAttention::new_ending_node(
                device,
                token_z,
                pairwise_head_width,
                pairwise_num_heads,
                1e9,
            ),
            transition_z: Transition::new(device, token_z, Some(token_z * 4), None),
            token_z,
            dropout,
        }
    }

    pub fn forward(
        &self,
        s: Tensor<B, 3>,
        z: Tensor<B, 4>,
        mask: Tensor<B, 3>,
        pair_mask: Tensor<B, 3>,
        chunk_size_tri_attn: Option<i64>,
        training: bool,
        _use_kernels: bool,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let mut z = z;
        let dropout = if training { self.dropout } else { 0.0 };

        for (tri, colwise) in [
            (self.tri_mul_out.forward(z.clone(), pair_mask.clone(), false), false),
            (self.tri_mul_in.forward(z.clone(), pair_mask.clone(), false), false),
            (
                self.tri_att_start
                    .forward(z.clone(), Some(pair_mask.clone()), chunk_size_tri_attn, false),
                false,
            ),
            (
                self.tri_att_end
                    .forward(z.clone(), Some(pair_mask.clone()), chunk_size_tri_attn, false),
                true,
            ),
        ] {
            let dm = if colwise {
                self.dropout_mask_columnwise(&z, dropout)
            } else {
                self.dropout_mask(&z, dropout)
            };
            z = z + dm * tri;
        }

        z = z.clone() + self.transition_z.forward(z, None);

        let s_normed = self.pre_norm_s.forward(s.clone());
        let s_out = self
            .attention
            .forward(s_normed.clone(), z.clone(), mask, s_normed, None);
        let mut s = s + s_out;
        s = s.clone() + self.transition_s.forward(s, None);
        if let Some(ref pn) = self.s_post_norm {
            s = pn.forward(s);
        }
        (s, z)
    }

    fn dropout_mask(&self, z: &Tensor<B, 4>, dropout: f64) -> Tensor<B, 4> {
        if dropout == 0.0 {
            return Tensor::<B, 4>::ones([1, 1, 1, 1], &z.device());
        }
        let scale = 1.0 / (1.0 - dropout);
        let v = z.clone().slice([0..z.dims()[0], 0..z.dims()[1], 0..1, 0..1]);
        let thr = Tensor::<B, 4>::full([1, 1, 1, 1], dropout, &z.device());
        v.clone()
            .lower_equal(thr)
            .float()
            * scale
    }

    fn dropout_mask_columnwise(&self, z: &Tensor<B, 4>, dropout: f64) -> Tensor<B, 4> {
        if dropout == 0.0 {
            return Tensor::<B, 4>::ones([1, 1, 1, 1], &z.device());
        }
        let scale = 1.0 / (1.0 - dropout);
        let v = z.clone().slice([0..z.dims()[0], 0..1, 0..z.dims()[2], 0..1]);
        let thr = Tensor::<B, 4>::full([1, 1, 1, 1], dropout, &z.device());
        v.clone()
            .lower_equal(thr)
            .float()
            * scale
    }
}

#[derive(Module, Debug)]
pub struct PairformerModule<B: Backend> {
    layers: Vec<PairformerLayer<B>>,
}

impl<B: Backend> PairformerModule<B> {
    pub fn new(
        device: &Device<B>,
        token_s: usize,
        token_z: usize,
        num_blocks: usize,
        num_heads: usize,
        dropout: f64,
    ) -> Self {
        let layers = (0..num_blocks)
            .map(|_| {
                PairformerLayer::new(
                    device,
                    token_s,
                    token_z,
                    num_heads,
                    dropout,
                    32,
                    4,
                    false,
                )
            })
            .collect();
        Self { layers }
    }

    pub fn forward(
        &self,
        s: Tensor<B, 3>,
        z: Tensor<B, 4>,
        mask: Tensor<B, 3>,
        pair_mask: Tensor<B, 3>,
        chunk_size_tri_attn: Option<i64>,
        training: bool,
        use_kernels: bool,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let mut s = s;
        let mut z = z;
        for layer in &self.layers {
            let (s_out, z_out) = layer.forward(
                s,
                z,
                mask.clone(),
                pair_mask.clone(),
                chunk_size_tri_attn,
                training,
                use_kernels,
            );
            s = s_out;
            z = z_out;
        }
        (s, z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn pairformer_layer_shape() {
        let device = Default::default();
        let layer = PairformerLayer::<B>::new(&device, 32, 24, 4, 0.0, 32, 4, false);
        let s = Tensor::<B, 3>::random(
            [1, 8, 32],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let z = Tensor::<B, 4>::random(
            [1, 8, 8, 24],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let mask = Tensor::<B, 3>::ones([1, 8, 8], &device);
        let (s_out, z_out) = layer.forward(s, z, mask.clone(), mask, None, false, false);
        assert_eq!(s_out.dims(), [1, 8, 32]);
        assert_eq!(z_out.dims(), [1, 8, 8, 24]);
    }
}
