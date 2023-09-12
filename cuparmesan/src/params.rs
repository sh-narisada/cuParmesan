use concrete_core::prelude::{LweDimension, *};

pub struct Param {
    pub lwe_dimension: LweDimension,
    pub glwe_dimension: GlweDimension,
    pub polynomial_size: PolynomialSize,
    pub pbs_level: DecompositionLevelCount,
    pub pbs_base_log: DecompositionBaseLog,
    pub ks_level: DecompositionLevelCount,
    pub ks_base_log: DecompositionBaseLog,
}

#[allow(dead_code)]
pub const PARM90: Param = Param {
    lwe_dimension: LweDimension(560),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(1024),
    pbs_level: DecompositionLevelCount(2),
    pbs_base_log: DecompositionBaseLog(8),
    ks_level: DecompositionLevelCount(11),
    ks_base_log: DecompositionBaseLog(1),
};

#[allow(dead_code)]
pub const PARM112: Param = Param {
    lwe_dimension: LweDimension(665),
    glwe_dimension: GlweDimension(1),
    polynomial_size: PolynomialSize(1024),
    pbs_level: DecompositionLevelCount(3),
    pbs_base_log: DecompositionBaseLog(7),
    ks_level: DecompositionLevelCount(5),
    ks_base_log: DecompositionBaseLog(3),
};
