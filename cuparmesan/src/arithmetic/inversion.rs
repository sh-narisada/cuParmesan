use crate::init_cuda_lwe;
use crate::pbs;
use crate::structs;
use concrete_core::prelude::*;

#[allow(non_snake_case)]
pub fn inversion_naive(
    engines: &mut structs::Engines,
    params: &mut structs::Params,
    keys: &mut structs::Keys,
    d_cc: &mut CudaLweCiphertextVector64,
    d_ca: &CudaLweCiphertextVector64,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut d_buf = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key_after_ks,
        params.noise,
        LweCiphertextCount(1),
    )?;
    let mut d_1 = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount(1),
    )?;
    let mut d_2 = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount(1),
    )?;
    let mut d_3 = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount(1),
    )?;
    let mut d_c = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount(1),
    )?;

    let d_inv1__pi_5 = pbs::inv1__pi_5(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        params.pi,
        &mut params.param,
        params.n,
    )
    .expect("pbs::inv1__pi_5 failed.");
    let d_inv2__pi_5 = pbs::inv2__pi_5(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        params.pi,
        &mut params.param,
        params.n,
    )
    .expect("pbs::inv2__pi_5  failed.");

    for i in 0..params.n {
        engines
            .cuda_engine
            .discard_copy_at_lwe_ciphertext_vector(&mut d_1, &d_ca, i as u32)?;
        engines
            .cuda_engine
            .discard_add_lwe_ciphertext_vector(&mut d_2, &d_1, &d_c)?;
        pbs::pbs_discard(engines, keys, &mut d_3, &d_2, &d_inv1__pi_5, &mut d_buf)?;
        engines
            .cuda_engine
            .discard_set_at_lwe_ciphertext_vector(d_cc, &d_3, i as u32)?;
        engines
            .cuda_engine
            .fuse_add_lwe_ciphertext_vector(&mut d_2, &d_1)?;
        pbs::pbs_discard(engines, keys, &mut d_c, &d_2, &d_inv2__pi_5, &mut d_buf)?;
    }

    Ok(())
}

#[allow(non_snake_case)]
pub fn inversion_optimized(
    engines: &mut structs::Engines,
    params: &mut structs::Params,
    keys: &mut structs::Keys,
    d_cc: &mut CudaLweCiphertextVector64,
    d_ca: &CudaLweCiphertextVector64,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut d_buf2 = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key_after_ks,
        params.noise,
        LweCiphertextCount(2),
    )?;
    let mut d_1 = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount(1),
    )?;
    let mut d_2 = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount(1),
    )?;
    let mut d_c = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount(1),
    )?;
    let mut d_c2 = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount(2),
    )?;

    let d_inv__pi_5 = pbs::inv__pi_5(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        params.pi,
        &mut params.param,
        params.n,
    )
    .expect("pbs::inv__pi_5 failed.");

    for i in 0..params.n {
        engines
            .cuda_engine
            .discard_copy_at_lwe_ciphertext_vector(&mut d_1, &d_ca, i as u32)?;
        engines
            .cuda_engine
            .discard_add_lwe_ciphertext_vector(&mut d_2, &d_1, &d_c)?;
        engines
            .cuda_engine
            .discard_set_at_lwe_ciphertext_vector(&mut d_c2, &d_2, 0 as u32)?;
        engines
            .cuda_engine
            .fuse_add_lwe_ciphertext_vector(&mut d_2, &d_1)?;
        engines
            .cuda_engine
            .discard_set_at_lwe_ciphertext_vector(&mut d_c2, &d_2, 1 as u32)?;
        pbs::pbs(engines, keys, &mut d_c2, &d_inv__pi_5, &mut d_buf2)?;
        engines
            .cuda_engine
            .discard_copy_at_lwe_ciphertext_vector(&mut d_2, &d_c2, 0 as u32)?;
        engines
            .cuda_engine
            .discard_set_at_lwe_ciphertext_vector(d_cc, &d_2, i as u32)?;
        engines
            .cuda_engine
            .discard_copy_at_lwe_ciphertext_vector(&mut d_c, &d_c2, 1 as u32)?;
    }

    Ok(())
}
