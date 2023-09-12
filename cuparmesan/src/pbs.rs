use crate::structs;
use crate::utilities;
use crate::Param;
use concrete_core::prelude::*;
use std::error::Error;

fn eval_lut(
    default_engine: &mut DefaultEngine,
    cuda_engine: &mut CudaEngine,
    pi: usize,
    param: &mut Param,
    lut: [u64; 1 << (5 - 1)],
    n: usize,
) -> Result<CudaGlweCiphertextVector64, Box<dyn std::error::Error>> {
    let f_lut: Vec<f64> = lut.iter().map(|&x| x as f64).collect::<Vec<_>>();
    let h_lut = utilities::create_accum(
        |x| f_lut[x as usize],
        default_engine,
        pi,
        param.polynomial_size.0,
        param.glwe_dimension,
        n,
    )?;
    let d_lut: CudaGlweCiphertextVector64 = cuda_engine.convert_glwe_ciphertext_vector(&h_lut)?;

    Ok(d_lut)
}

#[allow(non_snake_case)]
pub fn id__pi_5(
    default_engine: &mut DefaultEngine,
    cuda_engine: &mut CudaEngine,
    pi: usize,
    param: &mut Param,
    n: usize,
) -> Result<CudaGlweCiphertextVector64, Box<dyn Error>> {
    eval_lut(
        default_engine,
        cuda_engine,
        pi,
        param,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1],
        n,
    )
}

#[allow(non_snake_case)]
pub fn l_id__pi_5(
    default_engine: &mut DefaultEngine,
    cuda_engine: &mut CudaEngine,
    pi: usize,
    param: &mut Param,
    n: usize,
) -> Result<CudaGlweCiphertextVector64, Box<dyn Error>> {
    eval_lut(
        default_engine,
        cuda_engine,
        pi,
        param,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1],
        (n + 1) * n / 2,
    )
}

#[allow(non_snake_case)]
pub fn f_4__pi_5(
    default_engine: &mut DefaultEngine,
    cuda_engine: &mut CudaEngine,
    pi: usize,
    param: &mut Param,
    n: usize,
) -> Result<CudaGlweCiphertextVector64, Box<dyn Error>> {
    eval_lut(
        default_engine,
        cuda_engine,
        pi,
        param,
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        n,
    )
}

#[allow(non_snake_case)]
pub fn l_f_4__pi_5(
    default_engine: &mut DefaultEngine,
    cuda_engine: &mut CudaEngine,
    pi: usize,
    param: &mut Param,
    n: usize,
) -> Result<CudaGlweCiphertextVector64, Box<dyn Error>> {
    eval_lut(
        default_engine,
        cuda_engine,
        pi,
        param,
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        (n + 1) * n / 2,
    )
}

#[allow(non_snake_case)]
pub fn and__pi_5(
    default_engine: &mut DefaultEngine,
    cuda_engine: &mut CudaEngine,
    pi: usize,
    param: &mut Param,
    n: usize,
) -> Result<CudaGlweCiphertextVector64, Box<dyn Error>> {
    eval_lut(
        default_engine,
        cuda_engine,
        pi,
        param,
        [0, 0, 31, 0, 1, 0, 0, 0, 0, 0, 0, 0, 31, 0, 1, 0],
        n * n,
    )
}

#[allow(non_snake_case)]
pub fn comp__pi_5(
    default_engine: &mut DefaultEngine,
    cuda_engine: &mut CudaEngine,
    pi: usize,
    param: &mut Param,
    n: usize,
) -> Result<CudaGlweCiphertextVector64, Box<dyn Error>> {
    eval_lut(
        default_engine,
        cuda_engine,
        pi,
        param,
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        n,
    )
}

#[allow(non_snake_case)]
pub fn comp2__pi_5(
    default_engine: &mut DefaultEngine,
    cuda_engine: &mut CudaEngine,
    pi: usize,
    param: &mut Param,
    n: usize,
) -> Result<CudaGlweCiphertextVector64, Box<dyn Error>> {
    eval_lut(
        default_engine,
        cuda_engine,
        pi,
        param,
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        n,
    )
}

#[allow(non_snake_case)]
pub fn max__pi_5(
    default_engine: &mut DefaultEngine,
    cuda_engine: &mut CudaEngine,
    pi: usize,
    param: &mut Param,
    n: usize,
) -> Result<CudaGlweCiphertextVector64, Box<dyn Error>> {
    eval_lut(
        default_engine,
        cuda_engine,
        pi,
        param,
        [0, 31, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        2 * n,
    )
}

#[allow(non_snake_case)]
pub fn inv1__pi_5(
    default_engine: &mut DefaultEngine,
    cuda_engine: &mut CudaEngine,
    pi: usize,
    param: &mut Param,
    _n: usize,
) -> Result<CudaGlweCiphertextVector64, Box<dyn Error>> {
    eval_lut(
        default_engine,
        cuda_engine,
        pi,
        param,
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31],
        1,
    )
}

#[allow(non_snake_case)]
pub fn inv2__pi_5(
    default_engine: &mut DefaultEngine,
    cuda_engine: &mut CudaEngine,
    pi: usize,
    param: &mut Param,
    _n: usize,
) -> Result<CudaGlweCiphertextVector64, Box<dyn Error>> {
    eval_lut(
        default_engine,
        cuda_engine,
        pi,
        param,
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 31],
        1,
    )
}

#[allow(non_snake_case)]
pub fn inv__pi_5(
    default_engine: &mut DefaultEngine,
    cuda_engine: &mut CudaEngine,
    pi: usize,
    param: &mut Param,
    _n: usize,
) -> Result<CudaGlweCiphertextVector64, Box<dyn std::error::Error>> {
    let inv1__pi_5: Vec<f64> = vec![0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31]
        .iter()
        .map(|&x| x as f64)
        .collect::<Vec<_>>();
    let inv2__pi_5: Vec<f64> = vec![0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 31]
        .iter()
        .map(|&x| x as f64)
        .collect::<Vec<_>>();

    let lut1 =
        utilities::create_lut_vector(|x| inv1__pi_5[x as usize], pi, param.polynomial_size.0)?;
    let lut2 =
        utilities::create_lut_vector(|x| inv2__pi_5[x as usize], pi, param.polynomial_size.0)?;
    let h_inv__pi_5 =
        utilities::create_accum_from_two_lut(lut1, lut2, default_engine, param.glwe_dimension, 2)?;
    let d_inv__pi_5: CudaGlweCiphertextVector64 =
        cuda_engine.convert_glwe_ciphertext_vector(&h_inv__pi_5)?;

    Ok(d_inv__pi_5)
}

#[allow(non_snake_case)]
pub fn pbs(
    engines: &mut structs::Engines,
    keys: &mut structs::Keys,
    d_ca: &mut CudaLweCiphertextVector64,
    d_lut: &CudaGlweCiphertextVector64,
    d_buffer_lwe: &mut CudaLweCiphertextVector64,
) -> Result<(), Box<dyn std::error::Error>> {
    engines
        .cuda_engine
        .discard_keyswitch_lwe_ciphertext_vector(d_buffer_lwe, d_ca, keys.d_ksk)?;
    engines
        .cuda_amortized_engine
        .discard_bootstrap_lwe_ciphertext_vector(d_ca, d_buffer_lwe, d_lut, keys.d_bsk)?;

    Ok(())
}

#[allow(non_snake_case)]
pub fn pbs_discard(
    engines: &mut structs::Engines,
    keys: &mut structs::Keys,
    d_cb: &mut CudaLweCiphertextVector64,
    d_ca: &CudaLweCiphertextVector64,
    d_lut: &CudaGlweCiphertextVector64,
    d_buffer_lwe: &mut CudaLweCiphertextVector64,
) -> Result<(), Box<dyn std::error::Error>> {
    engines
        .cuda_engine
        .discard_keyswitch_lwe_ciphertext_vector(d_buffer_lwe, d_ca, keys.d_ksk)?;
    engines
        .cuda_amortized_engine
        .discard_bootstrap_lwe_ciphertext_vector(d_cb, d_buffer_lwe, d_lut, keys.d_bsk)?;

    Ok(())
}

#[allow(non_snake_case)]
pub fn pbs_k(
    engines: &mut structs::Engines,
    keys: &mut structs::Keys,
    d_ca: &mut CudaLweCiphertextVector64,
    d_lut: &CudaGlweCiphertextVector64,
    d_buffer_lwe: &mut CudaLweCiphertextVector64,
    k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    engines
        .cuda_engine
        .discard_keyswitch_lwe_ciphertext_vector_k(d_buffer_lwe, d_ca, keys.d_ksk, k)?;
    engines
        .cuda_amortized_engine
        .discard_bootstrap_lwe_ciphertext_vector_k(d_ca, d_buffer_lwe, d_lut, keys.d_bsk, k)?;

    Ok(())
}
