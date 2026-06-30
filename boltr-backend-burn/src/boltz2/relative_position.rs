//! Relative position encoding for pairwise features (`rel_pos`).

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::ElementConversion;
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Int, Tensor};

use crate::tensor_ops::{one_hot_int, where_int};

/// Index features for [`RelativePositionEncoder::forward`].
pub struct RelPosFeatures<'a, B: Backend> {
    pub asym_id: &'a Tensor<B, 2, Int>,
    pub residue_index: &'a Tensor<B, 2, Int>,
    pub entity_id: &'a Tensor<B, 2, Int>,
    pub token_index: &'a Tensor<B, 2, Int>,
    pub sym_id: &'a Tensor<B, 2, Int>,
    pub cyclic_period: &'a Tensor<B, 2, Int>,
}

#[derive(Module, Debug)]
pub struct RelativePositionEncoder<B: Backend> {
    linear_layer: Linear<B>,
    r_max: i64,
    s_max: i64,
    fix_sym_check: bool,
    cyclic_pos_enc: bool,
}

impl<B: Backend> RelativePositionEncoder<B> {
    pub fn new(
        device: &Device<B>,
        token_z: usize,
        r_max: Option<i64>,
        s_max: Option<i64>,
        fix_sym_check: bool,
        cyclic_pos_enc: bool,
    ) -> Self {
        let r_max = r_max.unwrap_or(32);
        let s_max = s_max.unwrap_or(2);
        let in_dim = 4 * (r_max + 1) + 2 * (s_max + 1) + 1;
        Self {
            linear_layer: LinearConfig::new(in_dim as usize, token_z)
                .with_bias(false)
                .init(device),
            r_max,
            s_max,
            fix_sym_check,
            cyclic_pos_enc,
        }
    }

    pub fn forward(&self, rel: RelPosFeatures<'_, B>) -> Tensor<B, 4> {
        let device = rel.asym_id.device();
        let b_same_chain = rel
            .asym_id
            .clone()
            .unsqueeze_dim::<3>(2)
            .equal(rel.asym_id.clone().unsqueeze_dim::<3>(1));
        let b_same_residue = rel
            .residue_index
            .clone()
            .unsqueeze_dim::<3>(2)
            .equal(rel.residue_index.clone().unsqueeze_dim::<3>(1));
        let b_same_entity = rel
            .entity_id
            .clone()
            .unsqueeze_dim::<3>(2)
            .equal(rel.entity_id.clone().unsqueeze_dim::<3>(1));

        let mut d_residue = rel.residue_index.clone().unsqueeze_dim::<3>(2)
            - rel.residue_index.clone().unsqueeze_dim::<3>(1);

        if self.cyclic_pos_enc {
            let pos = rel.cyclic_period.clone().greater_elem(0i64);
            let any_pos: i64 = pos.clone().int().sum().into_scalar().elem();
            if any_pos != 0 {
                let ten_k = Tensor::<B, 2, Int>::full(rel.cyclic_period.dims(), 10_000, &device);
                let period = ten_k.mask_where(pos, rel.cyclic_period.clone());
                let d_f = d_residue.clone().float();
                let p_f = period.unsqueeze_dim::<3>(2).float();
                let adj = d_f.clone() / p_f.clone();
                d_residue = (d_f - p_f * adj.round()).int();
            }
        }

        let r2 = 2 * self.r_max;
        let mut d_residue = (d_residue + self.r_max).clamp(0, r2);
        let off_chain = Tensor::<B, 3, Int>::full(d_residue.dims(), r2 + 1, &device);
        d_residue = where_int(b_same_chain.clone(), d_residue, off_chain);
        let a_rel_pos = one_hot_int(d_residue, (r2 + 2) as usize);

        let mut d_token = (rel.token_index.clone().unsqueeze_dim::<3>(2)
            - rel.token_index.clone().unsqueeze_dim::<3>(1)
            + self.r_max)
            .clamp(0, r2);
        let same_res = b_same_chain.clone().bool_and(b_same_residue);
        let off_tok = Tensor::<B, 3, Int>::full(d_token.dims(), r2 + 1, &device);
        d_token = where_int(same_res, d_token, off_tok);
        let a_rel_token = one_hot_int(d_token, (r2 + 2) as usize);

        let mut d_chain =
            (rel.sym_id.clone().unsqueeze_dim::<3>(2) - rel.sym_id.clone().unsqueeze_dim::<3>(1) + self.s_max)
                .clamp(0, 2 * self.s_max);
        let s2 = 2 * self.s_max;
        let cond = if self.fix_sym_check {
            b_same_entity.clone().bool_not()
        } else {
            b_same_chain.clone()
        };
        let off_ch = Tensor::<B, 3, Int>::full(d_chain.dims(), s2 + 1, &device);
        d_chain = where_int(cond, off_ch, d_chain);
        let a_rel_chain = one_hot_int(d_chain, (s2 + 2) as usize);

        let b_ent_f = b_same_entity.clone().float().unsqueeze_dim::<4>(3);
        let cat = Tensor::cat(
            vec![
                a_rel_pos,
                a_rel_token,
                b_ent_f,
                a_rel_chain,
            ],
            3,
        );
        self.linear_layer.forward(cat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn rel_pos_shape() {
        let device = Default::default();
        let b = 2;
        let n = 11;
        let token_z = 64;
        let enc = RelativePositionEncoder::<B>::new(&device, token_z, None, None, false, false);
        let asym_id = Tensor::<B, 2, Int>::zeros([b, n], &device);
        let residue_index =
            Tensor::<B, 1, Int>::arange(0..n as i64, &device).reshape([1, n]).repeat(&[b, 1]);
        let entity_id = Tensor::<B, 2, Int>::zeros([b, n], &device);
        let token_index = residue_index.clone();
        let sym_id = Tensor::<B, 2, Int>::zeros([b, n], &device);
        let cyclic_period = Tensor::<B, 2, Int>::zeros([b, n], &device);
        let rel = RelPosFeatures {
            asym_id: &asym_id,
            residue_index: &residue_index,
            entity_id: &entity_id,
            token_index: &token_index,
            sym_id: &sym_id,
            cyclic_period: &cyclic_period,
        };
        let z = enc.forward(rel);
        assert_eq!(z.dims(), [b, n, n, token_z]);
    }
}
