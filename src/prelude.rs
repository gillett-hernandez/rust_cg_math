pub(crate) use std::f32::INFINITY;
pub(crate) use std::f32::consts::{FRAC_PI_2, PI, TAU};
pub(crate) use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg};

pub use thermite::Vector;
pub use thermite::prelude::*;

pub use crate::bounds::*;
pub use crate::color::*;
// The dimensional algebra: re-export the combinators/trait and the composed
// `*Dim` aliases, but NOT the bare base structs (`dimension::{Length, SolidAngle,
// …}`), which deliberately shadow the measure names in `traits` — refer to those
// as `dimension::Length<N>` etc. to avoid a glob collision.
pub use crate::dimension::{
    AreaDim, BsdfDim, Dimension, Dimensionless, DimensionlessDim, IrradianceDim, IsDimensionless,
    LengthDim, Nil, Normalize, Normalized, PowerDim, Product, RadianceDim, Recip, SameDimension,
    SolidAngleDim, WavelengthDim,
};
pub use crate::dual::{Dual, SampleField, reciprocal_det_3, reciprocal_gram_det_2};
pub use crate::misc::*;
pub use crate::pdf::*;
pub use crate::point::Point3;
pub use crate::quantity::*;
pub use crate::random::*;
pub use crate::ray::*;
pub use crate::sample::*;
pub use crate::spectral::{
    HeroWavelength, SingleWavelength, WavelengthEnergy, WavelengthEnergyTrait,
};
pub use crate::traits::*;

pub use crate::curves::{
    Curve, CurveWithCDF, InterpolationMode, SpectralPowerDistributionFunction,
};

pub use crate::tangent_frame::TangentFrame;
pub use crate::transform::*;
pub use crate::vec::{Axis, Vec3};
