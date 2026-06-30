//! Trunk slice for Boltz2: init + recycling + `PairformerModule`.

use burn::module::Module;
use burn::nn::{LayerNorm, Linear};
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Tensor};

use crate::burn_compat::{layer_norm_1d, linear_no_bias};
use crate::layers::PairformerModule;

use super::msa_module::{MsaFeatures, MsaModule};
use super::template_module::{TemplateFeatures, TemplateV2Module};

#[derive(Module, Debug)]
pub struct TrunkV2<B: Backend> {
    token_s: usize,
    token_z: usize,
    s_init: Linear<B>,
    z_init_1: Linear<B>,
    z_init_2: Linear<B>,
    s_norm: LayerNorm<B>,
    z_norm: LayerNorm<B>,
    s_recycle: Linear<B>,
    z_recycle: Linear<B>,
    pairformer: PairformerModule<B>,
    msa: MsaModule<B>,
    template: Option<TemplateV2Module<B>>,
}

impl<B: Backend> TrunkV2<B> {
    pub fn new(
        device: &Device<B>,
        token_s: Option<usize>,
        token_z: Option<usize>,
        num_blocks: Option<usize>,
        template_args: Option<(usize, usize)>,
    ) -> Self {
        let token_s = token_s.unwrap_or(384);
        let token_z = token_z.unwrap_or(128);
        let num_blocks = num_blocks.unwrap_or(4);

        let template = template_args.map(|(template_dim, template_blocks)| {
            TemplateV2Module::new(
                device,
                token_z,
                template_dim,
                template_blocks,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        });

        Self {
            token_s,
            token_z,
            s_init: linear_no_bias(device, token_s, token_s),
            z_init_1: linear_no_bias(device, token_s, token_z),
            z_init_2: linear_no_bias(device, token_s, token_z),
            s_norm: layer_norm_1d(device, token_s),
            z_norm: layer_norm_1d(device, token_z),
            s_recycle: linear_no_bias(device, token_s, token_s),
            z_recycle: linear_no_bias(device, token_z, token_z),
            pairformer: PairformerModule::new(device, token_s, token_z, num_blocks, 16, 0.25),
            msa: MsaModule::new(device, token_s, token_z, None, None, None, None, None, None, None),
            template,
        }
    }

    pub fn initialize(&self, s_inputs: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let s_init = self.s_init.forward(s_inputs.clone());
        let z_init_1 = self.z_init_1.forward(s_inputs.clone()).unsqueeze_dim::<4>(2);
        let z_init_2 = self.z_init_2.forward(s_inputs).unsqueeze_dim::<4>(1);
        (s_init, z_init_1 + z_init_2)
    }

    pub fn apply_recycling(
        &self,
        s_init: Tensor<B, 3>,
        z_init: Tensor<B, 4>,
        s_prev: Tensor<B, 3>,
        z_prev: Tensor<B, 4>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let s = s_init + self.s_recycle.forward(self.s_norm.forward(s_prev));
        let z = z_init + self.z_recycle.forward(self.z_norm.forward(z_prev));
        (s, z)
    }

    pub fn forward_pairformer(
        &self,
        s: Tensor<B, 3>,
        z: Tensor<B, 4>,
        mask: Tensor<B, 3>,
        pair_mask: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        self.pairformer
            .forward(s, z, mask, pair_mask, None, false, false)
    }

    pub fn forward_from_init(
        &self,
        s_init: Tensor<B, 3>,
        z_init: Tensor<B, 4>,
        token_pad_mask: Tensor<B, 2>,
        recycling_steps: Option<usize>,
        msa_feats: Option<&MsaFeatures<'_, B>>,
        template_feats: Option<&TemplateFeatures<'_, B>>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let recycling_steps = recycling_steps.unwrap_or(0);
        let [batch_size, num_tokens, _] = s_init.dims();
        let device = s_init.device();

        let mask = token_pad_mask.clone();
        let pair_mask = mask.clone().unsqueeze_dim::<3>(2) * mask.unsqueeze_dim::<3>(1);

        let mut s = Tensor::<B, 3>::zeros([batch_size, num_tokens, self.token_s], &device);
        let mut z = Tensor::<B, 4>::zeros(
            [batch_size, num_tokens, num_tokens, self.token_z],
            &device,
        );

        for _ in 0..=recycling_steps {
            let (s_recycled, z_recycled) = self.apply_recycling(
                s_init.clone(),
                z_init.clone(),
                s.clone(),
                z.clone(),
            );
            s = s_recycled;
            z = z_recycled;

            if let (Some(tmpl), Some(feats)) = (&self.template, template_feats) {
                z = z.clone() + tmpl.forward(z.clone(), feats, pair_mask.clone(), false);
            }
            z = self
                .msa
                .forward_trunk_step(z, s.clone(), msa_feats, false, None, false);

            let (s_new, z_new) = self.forward_pairformer(
                s.clone(),
                z.clone(),
                pair_mask.clone(),
                pair_mask.clone(),
            );
            s = s_new;
            z = z_new;
        }

        (s, z)
    }

    pub fn token_s(&self) -> usize {
        self.token_s
    }

    pub fn token_z(&self) -> usize {
        self.token_z
    }

    pub fn has_template(&self) -> bool {
        self.template.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn trunk_initialize_and_pairformer() {
        let device = Default::default();
        let token_s = 128;
        let token_z = 32;
        let trunk = TrunkV2::<B>::new(&device, Some(token_s), Some(token_z), Some(1), None);
        let s_inputs = Tensor::<B, 3>::random(
            [2, 20, token_s],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let (s_init, z_init) = trunk.initialize(s_inputs);
        assert_eq!(s_init.dims(), [2, 20, token_s]);
        assert_eq!(z_init.dims(), [2, 20, 20, token_z]);

        let mask = Tensor::<B, 3>::ones([2, 20, 20], &device);
        let (s_out, z_out) = trunk.forward_pairformer(s_init, z_init, mask.clone(), mask);
        assert_eq!(s_out.dims(), [2, 20, token_s]);
        assert_eq!(z_out.dims(), [2, 20, 20, token_z]);
    }

    #[test]
    fn trunk_forward_from_init() {
        let device = Default::default();
        let token_s = 64;
        let token_z = 24;
        let trunk = TrunkV2::<B>::new(&device, Some(token_s), Some(token_z), Some(1), None);
        let s_init = Tensor::<B, 3>::random(
            [1, 10, token_s],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let z_init = Tensor::<B, 4>::random(
            [1, 10, 10, token_z],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let pad = Tensor::<B, 2>::ones([1, 10], &device);
        let (s, z) = trunk.forward_from_init(s_init, z_init, pad, Some(1), None, None);
        assert_eq!(s.dims(), [1, 10, token_s]);
        assert_eq!(z.dims(), [1, 10, 10, token_z]);
    }
}
