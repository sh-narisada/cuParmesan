## cuParmesan: A GPU Implementation of Parmesan Library



### Specifications

* This software is a GPU implementation of the Parmesan library (https://github.com/fakub/parmesan/)  developed by Jakub Klemsa and Melek Önen.

* cuParmesan supports multiparallel homomorphic operations in GPU environmets for multi-byte integers on TFHE:

  * Addition
  * Scalar multiplication
  * Multiplication
  * Sign
  * Comparison
  * Max
  * ReLU
  * Conversion from redundant representation to binary representation

* This software is implemented using Parmesan, Concrete-core (https://github.com/zama-ai/concrete-core)  and Concrete-cuda (https://www.zama.ai/post/concrete-core-v1-0-gamma-with-gpu-acceleration) libraries developed by ZAMA Team.

   

### Dependencies

* This software requires crates (libraries) as follows to run: 
  * Concrete-core (1.0.1)
  * Concrete-cuda (0.1.1)
  * Colored  (2.0.0)


### How to Run

```bash
# clone cuParmesan
git clone https://github.com/sh-narisada/cuParmesan.git
cd cuParmesan

# clone concrete-core
git clone -b concrete-core-1.0.1 https://github.com/zama-ai/concrete-core.git
cd concrete-core

# apply a patch file
git apply ../diff.patch

# compile and execute
cd ../cuparmesan
RUSTFLAGS="-C target-cpu=native" cargo build --release
./target/release/cuparmesan
```


* Files are located as follows: 

```bash
cuParmesan
├── concrete-core
│   ├── concrete-core
│   │	  ├── src/*  
│   │	  └── Cargo.toml 
│   ├── concrete-cuda
│   │	  ├── cuda/* 
│   │	  ├── src/*  
│   │	  └── Cargo.toml 
│   └── Cargo.toml
└── cuparmesan
    ├── src/main.rs
    ├── target/release/cuparmesan 
    └── Cargo.toml
```

* Make sure that your machine supports CUDA operations (please check `nvidia-smi` and `nvcc` commands)

### An Example of Homomorphic Operartions

```shell-session
$ ./target/release/cuparmesan
Input number A
15
Input number B
6
Input operation:
1 : Full Adder
2 : Full Scalar Multiplier
3 : Full Multiplier
4 : Full Multiplier with Parallel Reduction
5 : Sign
6 : Comparison
7 : Max (Relu)
8 : Inversion (Naive)
9 : Inversion (Optimized)
1
...
c             =           21 :: PASS (exp. 21)
```

### Differences

* We added / modified these 52 files listed below from Concrete-cuda and Concrete-core libraries utilizing some functions implemented in Parmesan library.
  * In Concrete-core
    * concrete-core/src/backends/cuda/implementation/engines/cuda_amortized_engine/lwe_ciphertext_vector_k_discarding_bootstrap.rs
    * concrete-core/src/backends/cuda/implementation/engines/cuda_amortized_engine/mod.rs
    * concrete-core/src/backends/cuda/implementation/engines/cuda_engine/{
      lwe_ciphertext_vector_discarding_addition_local.rs, lwe_ciphertext_vector_discarding_and.rs, lwe_ciphertext_vector_discarding_copy_at.rs, lwe_ciphertext_vector_discarding_max.rs, lwe_ciphertext_vector_discarding_mult_by_const.rs, lwe_ciphertext_vector_discarding_opposite_inplace.rs, lwe_ciphertext_vector_discarding_rotation_all.rs, lwe_ciphertext_vector_discarding_rotation.rs, lwe_ciphertext_vector_discarding_shiftaddition.rs, lwe_ciphertext_vector_k_discarding_addition.rs, lwe_ciphertext_vector_k_discarding_keyswitch.rs, lwe_ciphertext_vector_pointer.rs, mod.rs
    }
    * concrete-core/src/backends/cuda/implementation/engines/cuda_engine/mod.rs
    * concrete-core/src/backends/cuda/private/crypto/bootstrap/mod.rs
    * concrete-core/src/backends/cuda/private/crypto/keyswitch/mod.rs
    * concrete-core/src/backends/cuda/private/crypto/lwe/list.rs
    * concrete-core/src/backends/cuda/private/device.rs
    * concrete-core/src/specification/engines/{
      lwe_ciphertext_vector_discarding_addition_local.rs, lwe_ciphertext_vector_discarding_and.rs, lwe_ciphertext_vector_discarding_copy_at.rs, lwe_ciphertext_vector_discarding_max.rs, lwe_ciphertext_vector_discarding_mult_by_const.rs, lwe_ciphertext_vector_discarding_opposite_inplace.rs, lwe_ciphertext_vector_discarding_rotation_all.rs, lwe_ciphertext_vector_discarding_rotation.rs, lwe_ciphertext_vector_discarding_shiftaddition.rs, lwe_ciphertext_vector_k_discarding_addition.rs, lwe_ciphertext_vector_k_discarding_bootstrap.rs, lwe_ciphertext_vector_pointer.rs, mod.rs
    }

  * In Concrete-cuda
    * concrete-cuda/cuda/include/{
      linear_algebra.h
      addition.cu, 
      addition.cuh,
      and.cu, 
      and.cuh,
      copy_at.cu, 
      copy_at.cuh,
      max.cu, 
      max.cuh,
      multbyconst.cu, 
      multbyconst.cuh,
      negation_inplace.cu, 
      negation_inplace.cuh,
      negation.cuh, 
      rotation.cu, 
      rotation.cuh,
      shiftaddition.cu, 
      shiftaddition.cuh
    }
    * concrete-cuda/cuda/src/cuda_bind.h


### License

* This software is licensed under AGPLv3.
* The license follows the stronger one of Parmesan (AGPLv3) and Concrete-core (BSD-3-Clause-Clear).



### Contact

* sh-narisada [a.t.] kddi.com

  