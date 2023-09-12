use crate::init_cuda_lwe;
use crate::naf_vec;
use crate::pbs;
use crate::structs;
use concrete_core::prelude::*;

#[allow(non_snake_case)]
pub fn scalar_mul(
    k: i128,
    engines: &mut structs::Engines,
    params: &mut structs::Params,
    keys: &mut structs::Keys,
    d_cc: &mut CudaLweCiphertextVector64,
    d_ca: &mut CudaLweCiphertextVector64,
    d_buffer_lwe: &mut CudaLweCiphertextVector64,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut d_z2 = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount(params.n),
    )?;

    let d_id__pi_5 = pbs::id__pi_5(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        params.pi,
        &mut params.param,
        params.n,
    )
    .expect("pbs::id__pi_5 failed.");
    let d_f_4__pi_5 = pbs::f_4__pi_5(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        params.pi,
        &mut params.param,
        params.n,
    )
    .expect("pbs::f_4__pi_5 failed.");

    if k < 0 {
        engines
            .cuda_engine
            .discard_opp_lwe_ciphertext_vector_inplace(d_ca)?;
    }
    let k_abs = k.abs() as u32;
    let k_vec = naf_vec(k_abs);

    for (i, ki) in k_vec.iter().enumerate() {
        if *ki != 0 {
            if *ki < 0 {
                engines
                    .cuda_engine
                    .discard_opp_lwe_ciphertext_vector_inplace(d_ca)?;
            }
            engines
                .cuda_engine
                .discard_rotate_lwe_ciphertext_vector(&mut d_z2, &d_ca, i as u32)?;
            inplace_add(
                engines,
                params,
                keys,
                d_cc,
                &d_z2,
                &d_f_4__pi_5,
                d_buffer_lwe,
            )?;
            if *ki < 0 {
                engines
                    .cuda_engine
                    .discard_opp_lwe_ciphertext_vector_inplace(d_ca)?;
            }
            pbs::pbs(engines, keys, d_cc, &d_id__pi_5, d_buffer_lwe)?;
        }
    }

    Ok(())
}

#[allow(non_snake_case)]
fn inplace_add(
    engines: &mut structs::Engines,
    params: &mut structs::Params,
    keys: &mut structs::Keys,
    d_ca: &mut CudaLweCiphertextVector64,
    d_cb: &CudaLweCiphertextVector64,
    d_f_4__pi_5: &CudaGlweCiphertextVector64,
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
        .discard_bootstrap_lwe_ciphertext_vector(&mut d_q, d_buffer_lwe, d_f_4__pi_5, keys.d_bsk)?;
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
        .discard_sftadd_lwe_ciphertext_vector(d_ca, &d_z, &d_q)?;

    Ok(())
}

// 修正中
#[allow(non_snake_case)]
pub fn multiplication(
    engines: &mut structs::Engines,
    params: &mut structs::Params,
    keys: &mut structs::Keys,
    d_cc: &mut CudaLweCiphertextVector64,
    d_ca: &mut CudaLweCiphertextVector64,
    d_cb: &CudaLweCiphertextVector64,
    d_buffer_lwe: &mut CudaLweCiphertextVector64,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut d_lbuffer_lwe = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key_after_ks,
        params.noise,
        LweCiphertextCount(params.n * params.n),
    )?;
    let mut d_z2 = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount(params.n),
    )?;
    let mut d_3xy = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount(params.n * params.n),
    )?;
    let mut d_3xy_i = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount(params.n),
    )?;

    let d_and__pi_5 = pbs::and__pi_5(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        params.pi,
        &mut params.param,
        params.n,
    )
    .expect("pbs::and__pi_5 failed.");
    let d_f_4__pi_5 = pbs::f_4__pi_5(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        params.pi,
        &mut params.param,
        params.n,
    )
    .expect("pbs::f_4__pi_5 failed.");
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
        .discard_and_lwe_ciphertext_vector(&mut d_3xy, &d_ca, &d_cb)?;
    pbs::pbs(engines, keys, &mut d_3xy, &d_and__pi_5, &mut d_lbuffer_lwe)?;
    for i in 0..params.n {
        engines.cuda_engine.discard_copy_n_at_lwe_ciphertext_vector(
            &mut d_3xy_i,
            &d_3xy,
            i as u32,
        )?;
        engines
            .cuda_engine
            .discard_rotate_lwe_ciphertext_vector(&mut d_z2, &d_3xy_i, i as u32)?;
        inplace_add(
            engines,
            params,
            keys,
            d_cc,
            &d_z2,
            &d_f_4__pi_5,
            d_buffer_lwe,
        )?;
        pbs::pbs(engines, keys, d_cc, &d_id__pi_5, d_buffer_lwe)?;
    }

    Ok(())
}

#[allow(non_snake_case)]
pub fn mul_parallel(
    engines: &mut structs::Engines,
    params: &mut structs::Params,
    keys: &mut structs::Keys,
    d_cc: &mut CudaLweCiphertextVector64,
    d_ca: &mut CudaLweCiphertextVector64,
    d_cb: &CudaLweCiphertextVector64,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut d_lbuffer_lwe = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key_after_ks,
        params.noise,
        LweCiphertextCount(params.n * params.n),
    )?;
    let mut dl_buffer_lwe = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key_after_ks,
        params.noise,
        LweCiphertextCount((params.n + 1) * params.n / 2),
    )?;
    let mut d_3xy = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount(params.n * params.n),
    )?;
    let mut d_3xy_rot = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount(params.n * params.n),
    )?;
    let mut dl_w = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount((params.n + 1) * params.n / 2),
    )?;
    let mut dl_3w = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount((params.n + 1) * params.n / 2),
    )?;
    let mut dl_q = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount((params.n + 1) * params.n / 2),
    )?;
    let mut dl_2q = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount((params.n + 1) * params.n / 2),
    )?;
    let mut dl_z = init_cuda_lwe(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        &keys.h_secret_key,
        params.noise,
        LweCiphertextCount((params.n + 1) * params.n / 2),
    )?;

    let d_and__pi_5 = pbs::and__pi_5(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        params.pi,
        &mut params.param,
        params.n,
    )
    .expect("pbs::and__pi_5 failed.");
    let dl_f_4__pi_5 = pbs::l_f_4__pi_5(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        params.pi,
        &mut params.param,
        params.n,
    )
    .expect("pbs::l_f_4__pi_5 failed.");
    let dl_id__pi_5 = pbs::l_id__pi_5(
        &mut engines.default_engine,
        &mut engines.cuda_engine,
        params.pi,
        &mut params.param,
        params.n,
    )
    .expect("pbs::l_id__pi_5  failed.");

    engines
        .cuda_engine
        .discard_and_lwe_ciphertext_vector(&mut d_3xy, &d_ca, &d_cb)?;
    pbs::pbs(engines, keys, &mut d_3xy, &d_and__pi_5, &mut d_lbuffer_lwe)?;
    engines
        .cuda_engine
        .discard_rotate_all_lwe_ciphertext_vector(&mut d_3xy_rot, &d_3xy)?;

    let d: u32 = (params.n as f64).log2() as u32;
    for i in (0..d).rev() {
        engines
            .cuda_engine
            .discard_add_to_local_lwe_ciphertext_vector(&mut dl_w, &d_3xy_rot, i as u32, d - 1)?;
        engines
            .cuda_engine
            .discard_mult_by_const_lwe_ciphertext_vector_k(
                &mut dl_3w,
                &dl_w,
                3,
                (params.n + 1) * 2_i32.pow(i) as usize,
            )?;
        engines.cuda_engine.discard_sftadd_lwe_ciphertext_vector_k(
            &mut dl_q,
            &dl_3w,
            &dl_w,
            ((params.n + 1) as u32) * 2_i32.pow(i) as u32,
        )?;
        pbs::pbs_k(
            engines,
            keys,
            &mut dl_q,
            &dl_f_4__pi_5,
            &mut dl_buffer_lwe,
            (params.n + 1) * 2_i32.pow(i) as usize,
        )?;
        engines
            .cuda_engine
            .discard_mult_by_const_lwe_ciphertext_vector_k(
                &mut dl_2q,
                &dl_q,
                2,
                (params.n + 1) * 2_i32.pow(i) as usize,
            )?;
        engines
            .cuda_engine
            .discard_opp_lwe_ciphertext_vector_k_inplace(
                &mut dl_2q,
                (params.n + 1) * 2_i32.pow(i) as usize,
            )?;
        engines.cuda_engine.discard_add_lwe_ciphertext_vector_k(
            &mut dl_z,
            &dl_2q,
            &dl_w,
            (params.n + 1) * 2_i32.pow(i) as usize,
        )?;
        engines.cuda_engine.inplace_sftadd_lwe_ciphertext_vector_k(
            &mut dl_z,
            &dl_q,
            ((params.n + 1) as u32) * 2_i32.pow(i) as u32,
        )?;
        pbs::pbs_k(
            engines,
            keys,
            &mut dl_z,
            &dl_id__pi_5,
            &mut dl_buffer_lwe,
            (params.n + 1) * 2_i32.pow(i) as usize,
        )?;
        engines
            .cuda_engine
            .discard_copy_from_local_lwe_ciphertext_vector(
                &mut d_3xy_rot,
                &dl_z,
                i as u32,
                d - 1,
            )?;
    }

    *d_cc = d_3xy_rot;

    Ok(())
}
