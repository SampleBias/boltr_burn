//! Random-tensor smoke test for trunk → diffusion conditioning → sample → distogram.

use boltr_backend_burn::{
    AtomDiffusionConfig, Boltz2BurnModel, Boltz2BurnModelConfig, Boltz2DiffusionArgs,
    RelPosFeatures, SteeringParams,
};
use boltr_backend_core::Boltz2ModelDims;
use burn::backend::NdArray;
use burn::tensor::{Distribution, Int, Tensor};

type B = NdArray;

#[test]
fn predict_step_random_smoke_integration() {
    let device = Default::default();
    let config = Boltz2BurnModelConfig {
        dims: Boltz2ModelDims {
            token_s: 64,
            token_z: 32,
            num_pairformer_blocks: 1,
            bond_type_feature: false,
        },
        atom_s: 16,
        atom_z: 8,
        num_bins: 8,
    };
    let diff_args = Boltz2DiffusionArgs::tiny_for_tests(8);

    let model = Boltz2BurnModel::<B>::with_all_options(
        &device,
        &config,
        diff_args,
        AtomDiffusionConfig::default(),
        None,
        None,
        false,
    );

    let b = 1_usize;
    let n = 4_usize;
    let n_atoms = 4_usize;
    let token_s = model.token_s();
    let token_z = model.token_z();
    let num_bins = model.num_bins();

    let s_inputs = Tensor::<B, 3>::random(
        [b, n, token_s],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
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
    let pad = Tensor::<B, 2>::ones([b, n], &device);

    let (s_trunk, z_trunk) = model.forward_trunk_with_z_init_terms(
        s_inputs.clone(),
        &rel,
        None,
        None,
        pad.clone(),
        Some(0),
        None,
        None,
    );
    assert_eq!(s_trunk.dims(), [b, n, token_s]);
    assert_eq!(z_trunk.dims(), [b, n, n, token_z]);

    let rel_enc = model.forward_rel_pos(&rel);
    let ref_pos = Tensor::<B, 3>::random(
        [b, n_atoms, 3],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let ref_charge = Tensor::<B, 2>::random(
        [b, n_atoms],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let ref_element = Tensor::<B, 3>::random(
        [b, n_atoms, 4],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let atom_pad_mask = Tensor::<B, 2>::ones([b, n_atoms], &device);
    let ref_space_uid = Tensor::<B, 2, Int>::zeros([b, n_atoms], &device);
    let atom_to_token = Tensor::<B, 3>::zeros([b, n_atoms, n], &device);

    let cond = model.forward_diffusion_conditioning(
        s_trunk.clone(),
        z_trunk.clone(),
        rel_enc,
        ref_pos,
        ref_charge,
        ref_element,
        atom_pad_mask.clone(),
        ref_space_uid,
        atom_to_token.clone(),
        None,
    );

    let diffusion = model.forward_diffusion_sample(
        s_inputs,
        s_trunk,
        &cond,
        pad,
        atom_pad_mask,
        atom_to_token,
        Some(1),
        1,
        Some(SteeringParams::fast_path()),
    );
    assert_eq!(diffusion.sample_atom_coords.dims(), [1, n_atoms, 3]);

    let pdistogram = model.forward_distogram(z_trunk);
    assert_eq!(pdistogram.dims(), [b, n, n, 1, num_bins]);
}
