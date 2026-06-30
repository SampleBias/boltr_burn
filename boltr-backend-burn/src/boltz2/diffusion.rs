//! Structure module: `DiffusionModule` (score network) and `AtomDiffusion` (sampler).
//!
//! Reference: `boltz-reference/src/boltz/model/modules/diffusionv2.py`

use burn::module::Module;
use burn::nn::{LayerNorm, Linear};
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Int, Tensor};

use boltr_backend_core::Boltz2Hparams;

use crate::burn_compat::{layer_norm_1d, linear_no_bias};
use crate::tensor_ops::repeat_interleave_dim0;

use super::diffusion_conditioning::DiffusionConditioningOutput;
use super::encoders::{AtomAttentionDecoder, AtomAttentionEncoder, SingleConditioning};
use super::steering::SteeringParams;
use super::transformers::DiffusionTransformer;

// ---------------------------------------------------------------------------
// DiffusionModule  (score network)
// ---------------------------------------------------------------------------

/// The score model: atom encoder → token transformer → atom decoder.
#[derive(Module, Debug)]
pub struct DiffusionModule<B: Backend> {
    #[module(skip)]
    sigma_data: f64,
    single_conditioner: SingleConditioning<B>,
    atom_attention_encoder: AtomAttentionEncoder<B>,
    s_to_a_linear_norm: LayerNorm<B>,
    s_to_a_linear_linear: Linear<B>,
    token_transformer: DiffusionTransformer<B>,
    a_norm: LayerNorm<B>,
    atom_attention_decoder: AtomAttentionDecoder<B>,
    #[module(skip)]
    token_s: usize,
}

impl<B: Backend> DiffusionModule<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device<B>,
        token_s: usize,
        atom_s: usize,
        atoms_per_window_queries: usize,
        atoms_per_window_keys: usize,
        sigma_data: f64,
        dim_fourier: usize,
        atom_encoder_depth: usize,
        atom_encoder_heads: usize,
        token_transformer_depth: usize,
        token_transformer_heads: usize,
        atom_decoder_depth: usize,
        atom_decoder_heads: usize,
        conditioning_transition_layers: usize,
    ) -> Self {
        let two_ts = 2 * token_s;

        let single_conditioner = SingleConditioning::new(
            device,
            sigma_data,
            token_s,
            dim_fourier,
            conditioning_transition_layers,
            2,
        );

        let atom_attention_encoder = AtomAttentionEncoder::new(
            device,
            atom_s,
            token_s,
            atoms_per_window_queries,
            atoms_per_window_keys,
            atom_encoder_depth,
            atom_encoder_heads,
            true,
        );

        let s_to_a_linear_norm = layer_norm_1d(device, two_ts);
        let s_to_a_linear_linear = linear_no_bias(device, two_ts, two_ts);

        let token_transformer = DiffusionTransformer::new(
            device,
            token_transformer_depth,
            token_transformer_heads,
            two_ts,
            Some(two_ts),
            true,
        );

        let a_norm = layer_norm_1d(device, two_ts);

        let atom_attention_decoder = AtomAttentionDecoder::new(
            device,
            atom_s,
            token_s,
            atoms_per_window_queries,
            atoms_per_window_keys,
            atom_decoder_depth,
            atom_decoder_heads,
        );

        Self {
            sigma_data,
            single_conditioner,
            atom_attention_encoder,
            s_to_a_linear_norm,
            s_to_a_linear_linear,
            token_transformer,
            a_norm,
            atom_attention_decoder,
            token_s,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        s_inputs: Tensor<B, 3>,
        s_trunk: Tensor<B, 3>,
        r_noisy: Tensor<B, 3>,
        times: Tensor<B, 1>,
        cond: &DiffusionConditioningOutput<B>,
        token_pad_mask: Tensor<B, 2>,
        atom_pad_mask: Tensor<B, 2>,
        atom_to_token: Tensor<B, 3>,
        multiplicity: usize,
    ) -> Tensor<B, 3> {
        let s_trunk_rep = repeat_interleave_dim0(s_trunk, multiplicity);
        let s_inputs_rep = repeat_interleave_dim0(s_inputs, multiplicity);

        let (s, _normed_fourier) = self
            .single_conditioner
            .forward(times, s_trunk_rep, s_inputs_rep);

        let (a, q_skip, c_skip) = self.atom_attention_encoder.forward(
            cond.q.clone(),
            cond.c.clone(),
            cond.atom_enc_bias.clone(),
            atom_pad_mask.clone(),
            atom_to_token.clone(),
            r_noisy,
            multiplicity,
            &cond.indexing_matrix,
        );

        let s_to_a = self
            .s_to_a_linear_linear
            .forward(self.s_to_a_linear_norm.forward(s.clone()));
        let mut a = a + s_to_a;

        let mask = repeat_interleave_dim0(token_pad_mask, multiplicity);

        a = self.token_transformer.forward(
            a,
            s,
            Some(cond.token_trans_bias.clone()),
            mask,
            multiplicity,
            None,
        );
        a = self.a_norm.forward(a);

        self.atom_attention_decoder.forward(
            a,
            q_skip,
            c_skip,
            cond.atom_dec_bias.clone(),
            atom_pad_mask,
            atom_to_token,
            multiplicity,
            &cond.indexing_matrix,
        )
    }
}

// ---------------------------------------------------------------------------
// AtomDiffusion  (EDM sampler + preconditioning)
// ---------------------------------------------------------------------------

/// EDM-style diffusion sampler wrapping the score model.
#[derive(Module, Debug)]
pub struct AtomDiffusion<B: Backend> {
    pub score_model: DiffusionModule<B>,
    #[module(skip)]
    sigma_min: f64,
    #[module(skip)]
    sigma_max: f64,
    #[module(skip)]
    sigma_data: f64,
    #[module(skip)]
    rho: f64,
    #[module(skip)]
    num_sampling_steps: usize,
    #[module(skip)]
    gamma_0: f64,
    #[module(skip)]
    gamma_min: f64,
    #[module(skip)]
    noise_scale: f64,
    #[module(skip)]
    step_scale: f64,
    #[module(skip)]
    token_s: usize,
    #[module(skip)]
    alignment_reverse_diff: bool,
}

/// Configuration for constructing [`AtomDiffusion`].
pub struct AtomDiffusionConfig {
    pub num_sampling_steps: usize,
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub sigma_data: f64,
    pub rho: f64,
    pub gamma_0: f64,
    pub gamma_min: f64,
    pub noise_scale: f64,
    pub step_scale: f64,
    pub alignment_reverse_diff: bool,
}

impl Default for AtomDiffusionConfig {
    fn default() -> Self {
        Self {
            num_sampling_steps: 5,
            sigma_min: 0.0004,
            sigma_max: 160.0,
            sigma_data: 16.0,
            rho: 7.0,
            gamma_0: 0.8,
            gamma_min: 1.0,
            noise_scale: 1.003,
            step_scale: 1.5,
            alignment_reverse_diff: true,
        }
    }
}

impl AtomDiffusionConfig {
    /// Overlay `diffusion_process_args` from [`Boltz2Hparams`] when present.
    pub fn from_boltz2_hparams(h: &Boltz2Hparams) -> Self {
        let mut c = Self::default();
        let Some(v) = &h.diffusion_process_args else {
            return c;
        };
        let Some(obj) = v.as_object() else {
            return c;
        };
        if let Some(x) = obj.get("sigma_min").and_then(|x| x.as_f64()) {
            c.sigma_min = x;
        }
        if let Some(x) = obj.get("sigma_max").and_then(|x| x.as_f64()) {
            c.sigma_max = x;
        }
        if let Some(x) = obj.get("sigma_data").and_then(|x| x.as_f64()) {
            c.sigma_data = x;
        }
        if let Some(x) = obj.get("rho").and_then(|x| x.as_f64()) {
            c.rho = x;
        }
        if let Some(x) = obj.get("gamma_0").and_then(|x| x.as_f64()) {
            c.gamma_0 = x;
        }
        if let Some(x) = obj.get("gamma_min").and_then(|x| x.as_f64()) {
            c.gamma_min = x;
        }
        if let Some(x) = obj.get("noise_scale").and_then(|x| x.as_f64()) {
            c.noise_scale = x;
        }
        if let Some(x) = obj.get("step_scale").and_then(|x| x.as_f64()) {
            c.step_scale = x;
        }
        if let Some(x) = obj.get("alignment_reverse_diff").and_then(|x| x.as_bool()) {
            c.alignment_reverse_diff = x;
        }
        c
    }
}

/// Output of the diffusion sampling process.
pub struct DiffusionSampleOutput<B: Backend> {
    /// Denoised atom coordinates `[multiplicity, M, 3]`.
    pub sample_atom_coords: Tensor<B, 3>,
}

impl<B: Backend> AtomDiffusion<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device<B>,
        token_s: usize,
        atom_s: usize,
        atoms_per_window_queries: usize,
        atoms_per_window_keys: usize,
        dim_fourier: usize,
        atom_encoder_depth: usize,
        atom_encoder_heads: usize,
        token_transformer_depth: usize,
        token_transformer_heads: usize,
        atom_decoder_depth: usize,
        atom_decoder_heads: usize,
        conditioning_transition_layers: usize,
        config: AtomDiffusionConfig,
    ) -> Self {
        let score_model = DiffusionModule::new(
            device,
            token_s,
            atom_s,
            atoms_per_window_queries,
            atoms_per_window_keys,
            config.sigma_data,
            dim_fourier,
            atom_encoder_depth,
            atom_encoder_heads,
            token_transformer_depth,
            token_transformer_heads,
            atom_decoder_depth,
            atom_decoder_heads,
            conditioning_transition_layers,
        );

        Self {
            score_model,
            sigma_min: config.sigma_min,
            sigma_max: config.sigma_max,
            sigma_data: config.sigma_data,
            rho: config.rho,
            num_sampling_steps: config.num_sampling_steps,
            gamma_0: config.gamma_0,
            gamma_min: config.gamma_min,
            noise_scale: config.noise_scale,
            step_scale: config.step_scale,
            token_s,
            alignment_reverse_diff: config.alignment_reverse_diff,
        }
    }

    fn c_skip(&self, sigma: Tensor<B, 3>) -> Tensor<B, 3> {
        let sd2 = self.sigma_data * self.sigma_data;
        sd2 / (sigma.clone().powi_scalar(2) + sd2)
    }

    fn c_out(&self, sigma: Tensor<B, 3>) -> Tensor<B, 3> {
        sigma.clone() * self.sigma_data
            / (sigma.powi_scalar(2) + self.sigma_data * self.sigma_data).sqrt()
    }

    fn c_in(&self, sigma: Tensor<B, 3>) -> Tensor<B, 3> {
        1.0 / (sigma.powi_scalar(2) + self.sigma_data * self.sigma_data).sqrt()
    }

    fn c_noise(&self, sigma: Tensor<B, 1>) -> Tensor<B, 1> {
        (sigma / self.sigma_data).clamp_min(1e-20).log() * 0.25
    }

    #[allow(clippy::too_many_arguments)]
    fn preconditioned_network_forward(
        &self,
        noised_atom_coords: Tensor<B, 3>,
        sigma: Tensor<B, 1>,
        s_inputs: Tensor<B, 3>,
        s_trunk: Tensor<B, 3>,
        cond: &DiffusionConditioningOutput<B>,
        token_pad_mask: Tensor<B, 2>,
        atom_pad_mask: Tensor<B, 2>,
        atom_to_token: Tensor<B, 3>,
        multiplicity: usize,
    ) -> Tensor<B, 3> {
        let padded_sigma = sigma.clone().unsqueeze_dim::<3>(1);
        let r_noisy = self.c_in(padded_sigma.clone()) * noised_atom_coords.clone();
        let r_update = self.score_model.forward(
            s_inputs,
            s_trunk,
            r_noisy,
            self.c_noise(sigma),
            cond,
            token_pad_mask,
            atom_pad_mask,
            atom_to_token,
            multiplicity,
        );
        self.c_skip(padded_sigma.clone()) * noised_atom_coords + self.c_out(padded_sigma) * r_update
    }

    /// Karras et al. noise schedule: `sigma_i` from `sigma_max` to `sigma_min`.
    pub fn sample_schedule(&self, num_steps: usize, device: &Device<B>) -> Tensor<B, 1> {
        let inv_rho = 1.0 / self.rho;
        let steps = Tensor::<B, 1, Int>::arange(0..num_steps as i64, device).float();
        let scale =
            (self.sigma_min.powf(inv_rho) - self.sigma_max.powf(inv_rho)) / ((num_steps - 1) as f64);
        let sigmas = (steps.clone() * scale + self.sigma_max.powf(inv_rho)).powf_scalar(self.rho);
        let sigmas = sigmas * self.sigma_data;
        let zero = Tensor::<B, 1>::zeros([1], device);
        Tensor::cat(vec![sigmas, zero], 0)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &self,
        s_inputs: Tensor<B, 3>,
        s_trunk: Tensor<B, 3>,
        cond: &DiffusionConditioningOutput<B>,
        token_pad_mask: Tensor<B, 2>,
        atom_pad_mask: Tensor<B, 2>,
        atom_to_token: Tensor<B, 3>,
        num_sampling_steps: Option<usize>,
        multiplicity: usize,
    ) -> DiffusionSampleOutput<B> {
        self.sample_inner(
            s_inputs,
            s_trunk,
            cond,
            token_pad_mask,
            atom_pad_mask,
            atom_to_token,
            num_sampling_steps,
            multiplicity,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample_with_steering(
        &self,
        s_inputs: Tensor<B, 3>,
        s_trunk: Tensor<B, 3>,
        cond: &DiffusionConditioningOutput<B>,
        token_pad_mask: Tensor<B, 2>,
        atom_pad_mask: Tensor<B, 2>,
        atom_to_token: Tensor<B, 3>,
        num_sampling_steps: Option<usize>,
        multiplicity: usize,
        steering: &SteeringParams,
    ) -> DiffusionSampleOutput<B> {
        self.sample_inner(
            s_inputs,
            s_trunk,
            cond,
            token_pad_mask,
            atom_pad_mask,
            atom_to_token,
            num_sampling_steps,
            multiplicity,
            Some(steering),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_inner(
        &self,
        s_inputs: Tensor<B, 3>,
        s_trunk: Tensor<B, 3>,
        cond: &DiffusionConditioningOutput<B>,
        token_pad_mask: Tensor<B, 2>,
        atom_pad_mask: Tensor<B, 2>,
        atom_to_token: Tensor<B, 3>,
        num_sampling_steps: Option<usize>,
        multiplicity: usize,
        steering: Option<&SteeringParams>,
    ) -> DiffusionSampleOutput<B> {
        match steering {
            None => self.sample_fast(
                s_inputs,
                s_trunk,
                cond,
                token_pad_mask,
                atom_pad_mask,
                atom_to_token,
                num_sampling_steps,
                multiplicity,
            ),
            Some(s) if !s.uses_extended_sampler() => self.sample_fast(
                s_inputs,
                s_trunk,
                cond,
                token_pad_mask,
                atom_pad_mask,
                atom_to_token,
                num_sampling_steps,
                multiplicity,
            ),
            Some(_s) => self.sample_extended(
                s_inputs,
                s_trunk,
                cond,
                token_pad_mask,
                atom_pad_mask,
                atom_to_token,
                num_sampling_steps,
                multiplicity,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_fast(
        &self,
        s_inputs: Tensor<B, 3>,
        s_trunk: Tensor<B, 3>,
        cond: &DiffusionConditioningOutput<B>,
        token_pad_mask: Tensor<B, 2>,
        atom_pad_mask: Tensor<B, 2>,
        atom_to_token: Tensor<B, 3>,
        num_sampling_steps: Option<usize>,
        multiplicity: usize,
    ) -> DiffusionSampleOutput<B> {
        let device = s_trunk.device();
        let num_steps = num_sampling_steps.unwrap_or(self.num_sampling_steps);

        let atom_mask = repeat_interleave_dim0(atom_pad_mask.clone(), multiplicity);
        let [batch, atoms] = atom_mask.dims();
        let shape = [batch, atoms, 3];

        let sigmas = self.sample_schedule(num_steps, &device);

        let init_sigma = tensor_scalar_at(&sigmas, 0).unwrap_or(self.sigma_max * self.sigma_data);
        let mut atom_coords = Tensor::<B, 3>::random(
            shape,
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        ) * init_sigma;

        for step_idx in 0..num_steps {
            let sigma_tm = tensor_scalar_at(&sigmas, step_idx).unwrap();
            let sigma_t = tensor_scalar_at(&sigmas, step_idx + 1).unwrap();
            let gamma = if sigma_t > self.gamma_min {
                self.gamma_0
            } else {
                0.0
            };

            atom_coords = atom_coords.clone() - atom_coords.clone().mean_dim(1);

            let t_hat = sigma_tm * (1.0 + gamma);
            let noise_var =
                self.noise_scale * self.noise_scale * (t_hat * t_hat - sigma_tm * sigma_tm);
            let eps = Tensor::<B, 3>::random(
                shape,
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ) * noise_var.sqrt();
            let atom_coords_noisy = atom_coords.clone() + eps;

            let t_hat_tensor =
                Tensor::<B, 1>::full([batch], t_hat, &device);

            let atom_coords_denoised = self.preconditioned_network_forward(
                atom_coords_noisy.clone(),
                t_hat_tensor,
                s_inputs.clone(),
                s_trunk.clone(),
                cond,
                token_pad_mask.clone(),
                atom_pad_mask.clone(),
                atom_to_token.clone(),
                multiplicity,
            );

            let denoised_over_sigma =
                (atom_coords_noisy.clone() - atom_coords_denoised.clone()) / t_hat;
            atom_coords =
                atom_coords_noisy + (sigma_t - t_hat) * self.step_scale * denoised_over_sigma;
        }

        DiffusionSampleOutput {
            sample_atom_coords: atom_coords,
        }
    }

    /// Extended sampler with random augmentation, potentials guidance, and alignment.
    /// Potentials / FK resampling are not yet wired — falls back to fast path for now.
    #[allow(clippy::too_many_arguments)]
    fn sample_extended(
        &self,
        _s_inputs: Tensor<B, 3>,
        _s_trunk: Tensor<B, 3>,
        _cond: &DiffusionConditioningOutput<B>,
        _token_pad_mask: Tensor<B, 2>,
        _atom_pad_mask: Tensor<B, 2>,
        _atom_to_token: Tensor<B, 3>,
        _num_sampling_steps: Option<usize>,
        _multiplicity: usize,
    ) -> DiffusionSampleOutput<B> {
        todo!("extended diffusion sampler with potentials guidance not yet ported to Burn")
    }
}

fn tensor_scalar_at<B: Backend>(t: &Tensor<B, 1>, idx: usize) -> Option<f64> {
    let slice = t.clone().slice(idx..idx + 1).into_data();
    slice.as_slice::<f32>().ok().and_then(|s| s.first().map(|&v| v as f64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn sample_schedule_shape() {
        let device = Default::default();
        let cfg = AtomDiffusionConfig::default();
        let ad = AtomDiffusion::<B>::new(
            &device,
            32,
            16,
            4,
            8,
            64,
            1,
            2,
            1,
            2,
            1,
            2,
            2,
            cfg,
        );
        let sigmas = ad.sample_schedule(5, &device);
        assert_eq!(sigmas.dims(), [6]);
        let first = tensor_scalar_at(&sigmas, 0).unwrap();
        let last = tensor_scalar_at(&sigmas, 5).unwrap();
        assert!(first > 0.0, "first sigma should be positive");
        assert!(last.abs() < 1e-10, "last sigma should be ~0");
    }
}
