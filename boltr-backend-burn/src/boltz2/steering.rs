//! Inference-time steering / potentials configuration (Boltz `BoltzSteeringParams`).
//!
//! Reference: upstream Boltz `src/boltz/main.py` (`BoltzSteeringParams`, `predict` / `--use_potentials`).

/// Mirrors Python `BoltzSteeringParams` in upstream Boltz `src/boltz/main.py`.
#[derive(Debug, Clone, Copy)]
pub struct SteeringParams {
    pub fk_steering: bool,
    pub num_particles: i64,
    pub fk_lambda: f64,
    pub fk_resampling_interval: i64,
    pub physical_guidance_update: bool,
    pub contact_guidance_update: bool,
    pub num_gd_steps: i64,
}

impl Default for SteeringParams {
    fn default() -> Self {
        Self {
            fk_steering: false,
            num_particles: 3,
            fk_lambda: 4.0,
            fk_resampling_interval: 3,
            physical_guidance_update: false,
            contact_guidance_update: true,
            num_gd_steps: 20,
        }
    }
}

impl SteeringParams {
    /// Match Boltz CLI `predict`: `--use_potentials` sets `fk_steering` and `physical_guidance_update`.
    /// `contact_guidance_update` stays at the Python dataclass default (`true`).
    #[must_use]
    pub fn from_use_potentials(use_potentials: bool) -> Self {
        Self {
            fk_steering: use_potentials,
            physical_guidance_update: use_potentials,
            ..Self::default()
        }
    }

    /// Disable all steering (Rust fast path in `AtomDiffusion::sample` — no random aug / FK / guidance).
    #[must_use]
    pub fn fast_path() -> Self {
        Self {
            fk_steering: false,
            num_particles: 3,
            fk_lambda: 4.0,
            fk_resampling_interval: 3,
            physical_guidance_update: false,
            contact_guidance_update: false,
            num_gd_steps: 20,
        }
    }

    /// `true` when any of FK, physical guidance, or contact/template guidance is active.
    #[must_use]
    pub fn uses_extended_sampler(self) -> bool {
        self.fk_steering || self.physical_guidance_update || self.contact_guidance_update
    }

    /// `true` when `get_potentials(..., boltz2=True)` returns a non-empty list.
    #[must_use]
    pub fn needs_potential_list(self) -> bool {
        self.fk_steering || self.physical_guidance_update || (self.contact_guidance_update)
        // Boltz2 branch uses fk OR contact
    }
}
