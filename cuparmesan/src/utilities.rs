use concrete_core::prelude::{LweCiphertextCount, Variance, *};
use std::error::Error;

pub fn init_cuda_lwe(
    default_engine: &mut DefaultEngine,
    cuda_engine: &mut CudaEngine,
    s: &LweSecretKey64,
    noise: Variance,
    lwe_nums: LweCiphertextCount,
) -> Result<CudaLweCiphertextVector64, Box<dyn std::error::Error>> {
    let h_cipher: LweCiphertextVector64 =
        default_engine.zero_encrypt_lwe_ciphertext_vector(s, noise, lwe_nums)?;
    let d_cipher: CudaLweCiphertextVector64 =
        cuda_engine.convert_lwe_ciphertext_vector(&h_cipher)?;

    Ok(d_cipher)
}

pub fn descale_and_decode(
    msb_output: Vec<u64>,
    pi: usize,
    max: u64,
    two_pi: usize,
) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
    let enc_output = msb_output
        .iter()
        .map(|x| x >> (64 - pi - 1))
        .collect::<Vec<_>>();
    let carry = enc_output.iter().map(|x| x % 2).collect::<Vec<_>>();
    let output = enc_output
        .iter()
        .zip(carry.iter())
        .map(|(x, y)| ((x >> 1) + y) % (1 << (64 - pi)))
        .collect::<Vec<_>>();
    let output = output
        .iter()
        .map(|&x| {
            if x > max {
                x as i32 - two_pi as i32
            } else {
                x as i32
            }
        })
        .collect::<Vec<_>>();

    Ok(output)
}

pub fn create_accum<F>(
    func: F,
    default_engine: &mut DefaultEngine,
    bit_precision: usize,
    poly_size: usize,
    glwe_dim: GlweDimension,
    n: usize,
) -> Result<GlweCiphertextVector64, Box<dyn std::error::Error>>
where
    F: Fn(usize) -> f64,
{
    let delta = 1u64 << (64 - bit_precision);
    let mut accumulator_u64 = vec![0_u64; poly_size];
    let modulus_sup = 1 << (bit_precision - 1);
    let box_size = poly_size / modulus_sup;
    let half_box_size = box_size / 2;
    for i in 0..modulus_sup {
        let index = i as usize * box_size;
        accumulator_u64[index..index + box_size]
            .iter_mut()
            .for_each(|a| *a = (func(i) * delta as f64).round() as u64);
    }
    for a_i in accumulator_u64[0..half_box_size].iter_mut() {
        *a_i = (*a_i).wrapping_neg();
    }
    accumulator_u64.rotate_left(half_box_size);

    let mut accumulator_all: Vec<u64> = Vec::new();
    for _ in 0..n {
        accumulator_all.extend(&accumulator_u64);
    }
    let accumulator_plaintext = default_engine.create_plaintext_vector_from(&accumulator_all)?;
    let accumulator: GlweCiphertextVector64 = default_engine
        .trivially_encrypt_glwe_ciphertext_vector(
            glwe_dim.to_glwe_size(),
            GlweCiphertextCount(n),
            &accumulator_plaintext,
        )?;

    Ok(accumulator)
}

pub fn create_lut_vector<F>(
    func: F,
    bit_precision: usize,
    poly_size: usize,
) -> Result<Vec<u64>, Box<dyn std::error::Error>>
where
    F: Fn(usize) -> f64,
{
    let delta = 1u64 << (64 - bit_precision);
    let mut accumulator_u64 = vec![0_u64; poly_size];
    let modulus_sup = 1 << (bit_precision - 1);
    let box_size = poly_size / modulus_sup;
    let half_box_size = box_size / 2;
    for i in 0..modulus_sup {
        let index = i as usize * box_size;
        accumulator_u64[index..index + box_size]
            .iter_mut()
            .for_each(|a| *a = (func(i) * delta as f64).round() as u64);
    }
    for a_i in accumulator_u64[0..half_box_size].iter_mut() {
        *a_i = (*a_i).wrapping_neg();
    }
    accumulator_u64.rotate_left(half_box_size);

    Ok(accumulator_u64)
}

pub fn create_accum_from_two_lut(
    lut1: Vec<u64>,
    lut2: Vec<u64>,
    default_engine: &mut DefaultEngine,
    glwe_dim: GlweDimension,
    n: usize,
) -> Result<GlweCiphertextVector64, Box<dyn std::error::Error>> {
    let mut accumulator_all: Vec<u64> = Vec::new();
    accumulator_all.extend(&lut1);
    accumulator_all.extend(&lut2);
    let accumulator_plaintext = default_engine.create_plaintext_vector_from(&accumulator_all)?;
    let accumulator: GlweCiphertextVector64 = default_engine
        .trivially_encrypt_glwe_ciphertext_vector(
            glwe_dim.to_glwe_size(),
            GlweCiphertextCount(n),
            &accumulator_plaintext,
        )?;
    Ok(accumulator)
}

// this function was borrowed from parmesan
pub fn naf_vec(k: u32) -> Vec<i32> {
    if k == 0 {
        return vec![0];
    }
    if k == 1 {
        return vec![1];
    }
    let k_len = bit_len_32(k);
    let mut k_vec: Vec<i32> = Vec::new();
    let mut low_1: usize = 0;
    for i in 0..=k_len {
        k_vec.push(((k >> i) & 1) as i32);
        if (k >> i) & 1 == 0 {
            if i - low_1 > 1 {
                if low_1 > 0 && k_vec[low_1 - 1] == 1 {
                    k_vec[low_1 - 1] = -1;
                    k_vec[low_1] = 0;
                } else {
                    k_vec[low_1] = -1;
                }
                for j in low_1 + 1..i {
                    k_vec[j] = 0;
                }
                k_vec[i] = 1;
            }
            low_1 = i + 1;
        }
    }
    k_vec
}

// this function was borrowed from parmesan
pub fn convert_to_vec(m: i128, words: usize) -> Vec<i32> {
    let mut mv: Vec<i32> = Vec::new();
    let m_abs = m.abs();
    let m_pos = m >= 0;

    for i in 0..words {
        let mi = if ((m_abs >> i) & 1) == 0 {
            0i32
        } else {
            if m_pos {
                1i32
            } else {
                -1i32
            }
        };
        mv.push(mi);
    }

    mv
}

// this function was borrowed from parmesan
pub fn convert_from_vec(mv: &Vec<i32>) -> Result<i128, Box<dyn Error>> {
    let mut m = 0i128;
    for (i, mi) in mv.iter().enumerate() {
        m += match mi {
            1 => 1i128 << i,
            0 => 0i128,
            -1 => -(1i128 << i),
            _ => {
                return Err(format!("Word m_[{}] out of redundant bin alphabet: {}.", i, mi).into())
            }
        };
    }
    Ok(m)
}

// this function was borrowed from parmesan
#[inline]
pub fn bit_len_32(k: u32) -> usize {
    if k == 0 {
        return 0;
    }
    let mut k_len = 1;
    for i in 1..=31 {
        if k & (1 << i) != 0 {
            k_len = i + 1;
        }
    }
    k_len
}

#[allow(dead_code)]
pub fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}
