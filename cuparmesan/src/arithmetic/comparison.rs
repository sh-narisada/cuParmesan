use crate::arithmetic::addition;
use crate::init_cuda_lwe;
use crate::pbs;
use crate::structs;
use concrete_core::prelude::*;

#[allow(non_snake_case)]
pub fn sign(
    engines: &mut structs::Engines,
    params: &mut structs::Params,
    keys: &mut structs::Keys,
    d_cc: &mut CudaLweCiphertextVector64,
    d_ca: &mut CudaLweCiphertextVector64,
    d_buffer_lwe: &mut CudaLweCiphertextVector64,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut d_w = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount(params.n),
    )?;
    let d_comp__pi_5 = pbs::comp__pi_5(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        params.pi,
        &mut params.param,
        params.n,
    )
    .expect("pbs::comp__pi_5  failed.");
    
    let d: u32 = (params.n as f64).log2() as u32;
    for i in (0..d).rev() {
        engines
            .cuda_engine
            .discard_x_plus_2y_to_local_lwe_ciphertext_vector(&mut d_w, &d_ca, i as u32, d - 1)?;
        pbs::pbs_k(
            engines,
            keys,
            &mut d_w,
            &d_comp__pi_5,
            d_buffer_lwe,
            2_i32.pow(i) as usize,
        )?;
        engines
            .cuda_engine
            .discard_copy_from_local_for_sign_lwe_ciphertext_vector(d_ca, &d_w, i as u32, d - 1)?;
    }
    engines
        .cuda_engine
        .discard_copy_at_lwe_ciphertext_vector(d_cc, &d_ca, 0 as u32)?;

    Ok(())
}

#[allow(non_snake_case)]
pub fn comparison(
    engines: &mut structs::Engines,
    params: &mut structs::Params,
    keys: &mut structs::Keys,
    d_cc: &mut CudaLweCiphertextVector64,
    d_ca: &CudaLweCiphertextVector64,
    d_cb: &mut CudaLweCiphertextVector64,
    d_buffer_lwe: &mut CudaLweCiphertextVector64,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut d_w = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount(params.n),
    )?;
    let d_comp__pi_5 = pbs::comp__pi_5(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        params.pi,
        &mut params.param,
        params.n,
    )
    .expect("pbs::comp__pi_5  failed.");

    engines
        .cuda_engine
        .discard_opp_lwe_ciphertext_vector_inplace(d_cb)?;
    addition::addition(engines, params, keys, d_cc, &d_ca, &d_cb, d_buffer_lwe)?;
    let d: u32 = (params.n as f64).log2() as u32;
    for i in (0..d).rev() {
        engines
            .cuda_engine
            .discard_x_plus_2y_to_local_lwe_ciphertext_vector(&mut d_w, &d_cc, i as u32, d - 1)?;
        pbs::pbs_k(
            engines,
            keys,
            &mut d_w,
            &d_comp__pi_5,
            d_buffer_lwe,
            2_i32.pow(i) as usize,
        )?;
        engines
            .cuda_engine
            .discard_copy_from_local_for_sign_lwe_ciphertext_vector(d_cc, &d_w, i as u32, d - 1)?;
    }

    Ok(())
}

#[allow(non_snake_case)]
pub fn max(
    engines: &mut structs::Engines,
    params: &mut structs::Params,
    keys: &mut structs::Keys,
    d_cc: &mut CudaLweCiphertextVector64,
    d_ca: &CudaLweCiphertextVector64,
    d_cb: &mut CudaLweCiphertextVector64,
    d_buffer_lwe: &mut CudaLweCiphertextVector64,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut d_buf_xy = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key_after_ks,
        params.noise,
        LweCiphertextCount(2 * params.n),
    )?;
    let mut d_w = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount(params.n),
    )?;
    let mut d_1 = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount(1),
    )?;
    let mut d_xy = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount(2 * params.n),
    )?;
    let d_comp__pi_5 = pbs::comp__pi_5(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        params.pi,
        &mut params.param,
        params.n,
    )
    .expect("pbs::comp__pi_5  failed.");
    let d_comp2__pi_5 = pbs::comp2__pi_5(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        params.pi,
        &mut params.param,
        params.n,
    )
    .expect("pbs::comp2__pi_5  failed.");
    let d_max__pi_5 = pbs::max__pi_5(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        params.pi,
        &mut params.param,
        params.n,
    )
    .expect("pbs::max__pi_5  failed.");

    engines
        .cuda_engine
        .discard_opp_lwe_ciphertext_vector_inplace(d_cb)?;
    addition::addition(engines, params, keys, d_cc, &d_ca, &d_cb, d_buffer_lwe)?;
    let d: u32 = (params.n as f64).log2() as u32;
    for i in (0..d).rev() {
        engines
            .cuda_engine
            .discard_x_plus_2y_to_local_lwe_ciphertext_vector(&mut d_w, &d_cc, i as u32, d - 1)?;
        if i == 0 {
            pbs::pbs_k(
                engines,
                keys,
                &mut d_w,
                &d_comp2__pi_5,
                d_buffer_lwe,
                2_i32.pow(i) as usize,
            )?;
        } else {
            pbs::pbs_k(
                engines,
                keys,
                &mut d_w,
                &d_comp__pi_5,
                d_buffer_lwe,
                2_i32.pow(i) as usize,
            )?;
        }
        engines
            .cuda_engine
            .discard_copy_from_local_for_sign_lwe_ciphertext_vector(d_cc, &d_w, i as u32, d - 1)?;
    }
    engines
        .cuda_engine
        .discard_opp_lwe_ciphertext_vector_inplace(d_cb)?;
    engines
        .cuda_engine
        .discard_copy_at_lwe_ciphertext_vector(&mut d_1, &d_cc, 0 as u32)?;
    engines
        .cuda_engine
        .discard_extend_xy_lwe_ciphertext_vector(&mut d_xy, &d_ca, &d_cb, &d_1)?;
    pbs::pbs(engines, keys, &mut d_xy, &d_max__pi_5, &mut d_buf_xy)?;
    engines
        .cuda_engine
        .discard_merge_xy_lwe_ciphertext_vector(d_cc, &d_xy)?;

    Ok(())
}
