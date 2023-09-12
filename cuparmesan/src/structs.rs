use crate::params;
use concrete_core::prelude::*;

pub struct Engines<'a> {
    pub default_engine: &'a mut DefaultEngine,
    pub cuda_engine: &'a mut CudaEngine,
    pub cuda_amortized_engine: &'a mut AmortizedCudaEngine,
}

pub struct Params<'a> {
    pub param: &'a mut params::Param,
    pub pi: usize,
    pub n: usize,
    pub noise: Variance,
}

pub struct Keys<'a> {
    pub h_secret_key: &'a LweSecretKey64,
    pub h_secret_key_after_ks: &'a LweSecretKey64,
    pub d_bsk: &'a CudaFourierLweBootstrapKey64,
    pub d_ksk: &'a CudaLweKeyswitchKey64,
}
