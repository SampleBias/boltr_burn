//! Pairformer layer/module without sequence track.

use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Tensor};

use crate::layers::transition::Transition;
use crate::layers::triangular_attention::TriangleAttention;
use crate::layers::triangular_mult::{TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing};

fn dropout_mask_pair<B: Backend>(dropout: f64, z: &Tensor<B, 4>, training: bool) -> Tensor<B, 4> {
    if !training || dropout == 0.0 {
        return Tensor::<B, 4>::ones([1, 1, 1, 1], &z.device());
    }
    let scale = 1.0 / (1.0 - dropout);
    let v = z.clone().slice([0..z.dims()[0], 0..z.dims()[1], 0..1, 0..1]);
    let thr = Tensor::<B, 4>::full([1, 1, 1, 1], dropout, &z.device());
    v.lower_equal(thr).float() * scale
}

fn dropout_mask_columnwise<B: Backend>(
    dropout: f64,
    z: &Tensor<B, 4>,
    training: bool,
) -> Tensor<B, 4> {
    if !training || dropout == 0.0 {
        return Tensor::<B, 4>::ones([1, 1, 1, 1], &z.device());
    }
    let scale = 1.0 / (1.0 - dropout);
    let [b, i, j, c] = z.dims();
    let v = z.clone().slice([0..b, 0..1, 0..j, 0..1]);
    let thr = Tensor::<B, 4>::full([1, 1, 1, 1], dropout, &z.device());
    v.lower_equal(thr).float().repeat(&[b, i, 1, c]) * scale
}

#[derive(Module, Debug)]
pub struct PairformerNoSeqLayer<B: Backend> {
    dropout: f64,
    tri_mul_out: TriangleMultiplicationOutgoing<B>,
    tri_mul_in: TriangleMultiplicationIncoming<B>,
    tri_att_start: TriangleAttention<B>,
    tri_att_end: TriangleAttention<B>,
    transition_z: Transition<B>,
}

impl<B: Backend> PairformerNoSeqLayer<B> {
    pub fn new(
        device: &Device<B>,
        token_z: usize,
        dropout: f64,
        pairwise_head_width: usize,
        pairwise_num_heads: usize,
    ) -> Self {
        Self {
            dropout,
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
        }
    }

    pub fn forward(
        &self,
        z: Tensor<B, 4>,
        pair_mask: Tensor<B, 3>,
        chunk_size_tri_attn: Option<i64>,
        training: bool,
        use_kernels: bool,
    ) -> Tensor<B, 4> {
        let mut z = z;

        z = z.clone()
            + dropout_mask_pair(self.dropout, &z, training)
                * self.tri_mul_out.forward(z.clone(), pair_mask.clone(), use_kernels);
        z = z.clone()
            + dropout_mask_pair(self.dropout, &z, training)
                * self.tri_mul_in.forward(z.clone(), pair_mask.clone(), use_kernels);
        z = z.clone()
            + dropout_mask_pair(self.dropout, &z, training)
                * self.tri_att_start.forward(
                    z.clone(),
                    Some(pair_mask.clone()),
                    chunk_size_tri_attn,
                    use_kernels,
                );
        z = z.clone()
            + dropout_mask_columnwise(self.dropout, &z, training)
                * self.tri_att_end.forward(
                    z.clone(),
                    Some(pair_mask.clone()),
                    chunk_size_tri_attn,
                    use_kernels,
                );

        z.clone() + self.transition_z.forward(z, None)
    }
}

#[derive(Module, Debug)]
pub struct PairformerNoSeqModule<B: Backend> {
    layers: Vec<PairformerNoSeqLayer<B>>,
}

impl<B: Backend> PairformerNoSeqModule<B> {
    pub fn new(
        device: &Device<B>,
        token_z: usize,
        num_blocks: usize,
        dropout: f64,
        pairwise_head_width: usize,
        pairwise_num_heads: usize,
    ) -> Self {
        let layers = (0..num_blocks)
            .map(|_| {
                PairformerNoSeqLayer::new(
                    device,
                    token_z,
                    dropout,
                    pairwise_head_width,
                    pairwise_num_heads,
                )
            })
            .collect();
        Self { layers }
    }

    pub fn forward(
        &self,
        z: Tensor<B, 4>,
        pair_mask: Tensor<B, 3>,
        use_kernels: bool,
    ) -> Tensor<B, 4> {
        let n = z.dims()[1];
        let chunk_size = if n > 256 { Some(128) } else { Some(512) };
        let mut z = z;
        for layer in &self.layers {
            z = layer.forward(z, pair_mask.clone(), chunk_size, false, use_kernels);
        }
        z
    }

    pub fn num_blocks(&self) -> usize {
        self.layers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn pairformer_no_seq_shape() {
        let device = Default::default();
        let layer = PairformerNoSeqLayer::<B>::new(&device, 24, 0.0, 32, 4);
        let z = Tensor::<B, 4>::random(
            [1, 8, 8, 24],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let mask = Tensor::<B, 3>::ones([1, 8, 8], &device);
        let out = layer.forward(z, mask, None, false, false);
        assert_eq!(out.dims(), [1, 8, 8, 24]);
    }
}
