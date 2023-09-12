use crate::init_cuda_lwe;
use crate::pbs;
use crate::structs;
use concrete_core::prelude::*;

#[allow(non_snake_case)]
pub fn addition(
    engines: &mut structs::Engines,
    params: &mut structs::Params,
    keys: &mut structs::Keys,
    d_cc: &mut CudaLweCiphertextVector64,
    d_ca: &CudaLweCiphertextVector64,
    d_cb: &CudaLweCiphertextVector64,
    d_buffer_lwe: &mut CudaLweCiphertextVector64,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut d_w = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount(params.n),
    )?;
    let mut d_3w = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount(params.n),
    )?;
    let mut d_q = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount(params.n),
    )?;
    let mut d_2q = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount(params.n),
    )?;
    let mut d_z = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount(params.n),
    )?;

    let d_f_4__pi_5 = pbs::f_4__pi_5(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        params.pi,
        &mut params.param,
        params.n,
    )
    .expect("pbs::f_4_pi_5 failed.");
    let d_id__pi_5 = pbs::id__pi_5(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        params.pi,
        &mut params.param,
        params.n,
    )
    .expect("pbs::id__pi_5  failed.");

    engines
        .cuda_engine
        .discard_add_lwe_ciphertext_vector(&mut d_w, &d_ca, &d_cb)?;
    engines
        .cuda_engine
        .discard_mult_by_const_lwe_ciphertext_vector(&mut d_3w, &d_w, 3)?;
    engines
        .cuda_engine
        .discard_sftadd_lwe_ciphertext_vector(&mut d_q, &d_3w, &d_w)?;
    engines
        .cuda_engine
        .discard_keyswitch_lwe_ciphertext_vector(d_buffer_lwe, &mut d_q, keys.d_ksk)?;
    engines
        .cuda_amortized_engine
        .discard_bootstrap_lwe_ciphertext_vector(
            &mut d_q,
            d_buffer_lwe,
            &d_f_4__pi_5,
            keys.d_bsk,
        )?;
    engines
        .cuda_engine
        .discard_mult_by_const_lwe_ciphertext_vector(&mut d_2q, &d_q, 2)?;
    engines
        .cuda_engine
        .discard_opp_lwe_ciphertext_vector_inplace(&mut d_2q)?;
    engines
        .cuda_engine
        .discard_add_lwe_ciphertext_vector(&mut d_z, &d_2q, &d_w)?;
    engines
        .cuda_engine
        .discard_sftadd_lwe_ciphertext_vector(d_cc, &d_z, &d_q)?;
    pbs::pbs(engines, keys, d_cc, &d_id__pi_5, d_buffer_lwe)?;

    Ok(())
}
