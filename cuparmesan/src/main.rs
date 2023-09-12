/*
cuParmesan :  A GPU implementation of Parmesan using Concrete-core and Concrete-cuda libraries

- Dependencies
    Concrete-core (1.0.1)  :https://crates.io/crates/concrete-core/1.0.1 by Zama Team
    Concrete-cuda (0.1.1)  :https://crates.io/crates/concrete-cuda/0.1.1 by Zama Team
    Parmesan (0.0.20-alpha):https://crates.io/crates/parmesan/0.0.20-alpha by Jakub Klemsa and Melek Önen

- How to run:
cd cuparmesan
RUSTFLAGS="-C target-cpu=native" cargo build --release
./target/release/cuparmesan
*/

use colored::Colorize;
use concrete_core::prelude::{LweCiphertextCount, Variance, *};
use cuparmesan::{arithmetic::*, *};
use std::collections::HashMap;
use std::error::Error;
use std::io;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Input number A");
    let mut input1: String = String::new();
    std::io::stdin().read_line(&mut input1).ok();
    let a_val: i128 = input1.trim().parse().ok().unwrap();
    println!("Input number B");
    let mut input2: String = String::new();
    std::io::stdin().read_line(&mut input2).ok();
    let b_val: i128 = input2.trim().parse().ok().unwrap();

    let op_pairs = vec![
        (1, "Full Adder"),
        (2, "Full Scalar Multiplier"),
        (3, "Full Multiplier"),
        (4, "Full Multiplier with Parallel Reduction"),
        (5, "Sign"),
        (6, "Comparison"),
        (7, "Max (Relu)"),
        (8, "Inversion (Naive)"),
        (9, "Inversion (Optimized)"),
    ];
    let operation_map: HashMap<_, _> = op_pairs.into_iter().collect();
    println!("Input operation:");
    for i in 1..operation_map.len() + 1 {
        println!("{} : {}", i, operation_map.get(&i).unwrap());
    }
    let mut input3: String = String::new();
    io::stdin().read_line(&mut input3).ok();
    let operation: i32 = input3.trim().parse().ok().unwrap();

    let param_bit = 112;
    let mut param = if param_bit == 90 {
        params::PARM90
    } else {
        params::PARM112
    };
    let noise = Variance(2_f64.powf(-36.));
    let glwe_noise = if param_bit == 90 {
        Variance(2_f64.powf(-63.))
    } else {
        Variance(2_f64.powf(-56.))
    };
    println!("LWE Noise :{:?} \t GLWE Noise :{:?}", noise, glwe_noise);
    const PI: usize = 5; // PI-bit cleartext in range
    let max: u64 = 15; // 2^{PI - 1} - 1
    let two_pi: usize = 32; // 2^PI
    let n = 32; // # of bit
    let k: i128 = 121; // Constant for scalar multiplication

    const UNSAFE_SECRET: u128 = 0;
    let mut default_engine = DefaultEngine::new(Box::new(UnixSeeder::new(UNSAFE_SECRET)))?;
    let mut cuda_engine = CudaEngine::new(())?;
    let mut cuda_amortized_engine = AmortizedCudaEngine::new(())?;

    let a: Vec<i32> = convert_to_vec(a_val, n);
    let b: Vec<i32> = convert_to_vec(b_val, n);
    println!("n: {:?}", n);
    println!("a: {:?}", a);
    println!("b: {:?}", b);
    println!("a_val: {:?}", a_val);
    println!("b_val: {:?}", b_val);
    println!("k: {:?}", k);
    println!("k_vec: {:?}", convert_to_vec(k, 8));
    println!("log2(a): {:?}", (a_val as f64).log2());
    println!("log2(b): {:?}", (b_val as f64).log2());

    let mask: i32 = (1i32 << PI) - 1;
    let enc_a: Vec<u64> = a.iter().map(|x| (x & mask) as u64).collect::<Vec<_>>();
    let enc_b: Vec<u64> = b.iter().map(|x| (x & mask) as u64).collect::<Vec<_>>();
    println!("Encoded Input: {:?}", enc_a);

    let msb_a: Vec<u64> = enc_a.iter().map(|x| x << 64 - PI).collect::<Vec<_>>();
    let msb_b: Vec<u64> = enc_b.iter().map(|x| x << 64 - PI).collect::<Vec<_>>();

    println!("Encrypt...");
    let h_lut_key: GlweSecretKey64 =
        default_engine.generate_new_glwe_secret_key(param.glwe_dimension, param.polynomial_size)?;
    let h_secret_key: LweSecretKey64 =
        default_engine.transform_glwe_secret_key_to_lwe_secret_key(h_lut_key.clone())?;
    let h_secret_key_after_ks: LweSecretKey64 =
        default_engine.generate_new_lwe_secret_key(param.lwe_dimension)?;

    let h_a: PlaintextVector64 = default_engine.create_plaintext_vector_from(&msb_a)?;
    let h_b: PlaintextVector64 = default_engine.create_plaintext_vector_from(&msb_b)?;
    let h_ca: LweCiphertextVector64 =
        default_engine.encrypt_lwe_ciphertext_vector(&h_secret_key, &h_a, noise)?;
    let h_cb: LweCiphertextVector64 =
        default_engine.encrypt_lwe_ciphertext_vector(&h_secret_key, &h_b, noise)?;
    println!("# of LWE ciphertexts {:?}", h_ca.lwe_ciphertext_count());

    println!("Create Key Switch Key");
    let h_ksk: LweKeyswitchKey64 = default_engine.generate_new_lwe_keyswitch_key(
        &h_secret_key,
        &h_secret_key_after_ks,
        param.ks_level,
        param.ks_base_log,
        noise,
    )?;
    println!("Create Boostrap Key (it is very slow...)");
    let h_bsk: LweBootstrapKey64 = default_engine.generate_new_lwe_bootstrap_key(
        &h_secret_key_after_ks,
        &h_lut_key,
        param.pbs_base_log,
        param.pbs_level,
        glwe_noise,
    )?;

    let start = Instant::now();

    let d_bsk: CudaFourierLweBootstrapKey64 = cuda_engine.convert_lwe_bootstrap_key(&h_bsk)?;
    let d_ksk: CudaLweKeyswitchKey64 = cuda_engine.convert_lwe_keyswitch_key(&h_ksk)?;

    let mut d_buffer_lwe = init_cuda_lwe(
        &mut default_engine,
        &mut cuda_engine,
        &h_secret_key_after_ks,
        noise,
        LweCiphertextCount(n),
    )?;

    let mut d_ca: CudaLweCiphertextVector64 = cuda_engine.convert_lwe_ciphertext_vector(&h_ca)?;
    let mut d_cb: CudaLweCiphertextVector64 = cuda_engine.convert_lwe_ciphertext_vector(&h_cb)?;

    let mut d_cc = init_cuda_lwe(
        &mut default_engine,
        &mut cuda_engine,
        &h_secret_key,
        noise,
        LweCiphertextCount(n),
    )?;

    let end = start.elapsed();
    println!("{:4}ms : Malloc on GPU", end.as_millis());

    let mut engines: structs::Engines = structs::Engines {
        default_engine: &mut default_engine,
        cuda_engine: &mut cuda_engine,
        cuda_amortized_engine: &mut cuda_amortized_engine,
    };
    let mut keys: structs::Keys = structs::Keys {
        h_secret_key: &h_secret_key,
        h_secret_key_after_ks: &h_secret_key_after_ks,
        d_bsk: &d_bsk,
        d_ksk: &d_ksk,
    };
    let mut params: structs::Params = structs::Params {
        param: &mut param,
        pi: PI,
        n: n,
        noise: noise,
    };

    let mut result = 0;
    println!(
        "--- {} Start ---",
        operation_map.get(&(operation as usize)).unwrap()
    );
    let start = Instant::now();
    match operation {
        1 => {
            addition::addition(
                &mut engines,
                &mut params,
                &mut keys,
                &mut d_cc,
                &d_ca,
                &d_cb,
                &mut d_buffer_lwe,
            )?;
            result = a_val + b_val;
        }
        2 => {
            multiplication::scalar_mul(
                k,
                &mut engines,
                &mut params,
                &mut keys,
                &mut d_cc,
                &mut d_ca,
                &mut d_buffer_lwe,
            )?;
            result = a_val * k;
        }
        3 => {
            multiplication::multiplication(
                &mut engines,
                &mut params,
                &mut keys,
                &mut d_cc,
                &mut d_ca,
                &d_cb,
                &mut d_buffer_lwe,
            )?;
            result = a_val * b_val;
        }
        4 => {
            multiplication::mul_parallel(
                &mut engines,
                &mut params,
                &mut keys,
                &mut d_cc,
                &mut d_ca,
                &d_cb,
            )?;
            result = a_val * b_val;
        }
        5 => {
            comparison::sign(
                &mut engines,
                &mut params,
                &mut keys,
                &mut d_cc,
                &mut d_ca,
                &mut d_buffer_lwe,
            )?;
            result = if a_val > 0 {
                1
            } else if a_val == 0 {
                0
            } else {
                -1
            };
        }
        6 => {
            comparison::comparison(
                &mut engines,
                &mut params,
                &mut keys,
                &mut d_cc,
                &d_ca,
                &mut d_cb,
                &mut d_buffer_lwe,
            )?;
            result = if a_val > b_val {
                1
            } else if a_val == b_val {
                0
            } else {
                -1
            };
        }
        7 => {
            comparison::max(
                &mut engines,
                &mut params,
                &mut keys,
                &mut d_cc,
                &d_ca,
                &mut d_cb,
                &mut d_buffer_lwe,
            )?;
            result = if a_val >= b_val { a_val } else { b_val };
        }
        8 => {
            inversion::inversion_naive(&mut engines, &mut params, &mut keys, &mut d_cc, &d_ca)?;
            result = a_val;
        }
        9 => {
            inversion::inversion_optimized(&mut engines, &mut params, &mut keys, &mut d_cc, &d_ca)?;
            result = a_val;
        }
        _ => {}
    }
    let end = start.elapsed();
    println!(
        "{:4}ms : {} of {} bit redundant integers",
        end.as_millis(),
        operation_map.get(&(operation as usize)).unwrap(),
        n
    );
    println!(
        "--- {} End ---",
        operation_map.get(&(operation as usize)).unwrap()
    );

    let start = Instant::now();
    let h_cc: LweCiphertextVector64 = cuda_engine.convert_lwe_ciphertext_vector(&d_cc).unwrap();
    let end = start.elapsed();
    println!("{:4}us : Transfer GPU -> CPU", end.as_micros());

    let h_c: PlaintextVector64 =
        default_engine.decrypt_lwe_ciphertext_vector(&h_secret_key, &h_cc)?;
    let msb_c = default_engine.retrieve_plaintext_vector(&h_c)?;
    let c = descale_and_decode(msb_c, PI, max, two_pi)?;
    println!("c: {:?}", &c[..n]);

    let mut c_val = if operation == 5 || operation == 6 { // for sign and comparison
        c[0] as i128
    } else {
        convert_from_vec(&c[..n].to_vec())?
    };
    if c_val >= 2_i128.pow(n as u32 - 1) - 1 {
        c_val = c_val - 2_i128.pow(n as u32)
    }
    println!("c_val: {:?}", c_val);

    println!(
        "c             = {:12} :: {} (exp. {})",
        c_val,
        if c_val == result {
            String::from("PASS").bold().green()
        } else {
            String::from("FAIL").bold().red()
        },
        result
    );

    Ok(())
}
