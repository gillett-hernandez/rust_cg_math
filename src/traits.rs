use crate::prelude::*;
pub(crate) use std::ops::{Add, Div, Mul, Neg};
use std::{
    cmp::Ordering,
    fmt::Debug,
    ops::{AddAssign, MulAssign},
};

// differential forms of various measures
pub trait Measure: Copy + Clone + Debug {}

// differential solid angle
//      = sin(theta) d[theta] d[phi]
//      = d[cos theta] d[phi]
#[derive(Copy, Clone, Debug)]
pub struct SolidAngle {}
impl Measure for SolidAngle {}

// differential projected solid angle
//      = |W x N| * differential solid angle
//      = |cos(theta)| sin(theta) d[theta] d[phi]
//      = |cos(theta)| d[cos theta] d[phi]
//      = sin(theta) d[sin(theta)] dphi
#[derive(Copy, Clone, Debug)]
pub struct ProjectedSolidAngle {}
impl Measure for ProjectedSolidAngle {}

#[derive(Copy, Clone, Debug)]
pub struct Area {}
impl Measure for Area {}

// basic measure
#[derive(Copy, Clone, Debug)]
pub struct Uniform01 {}
impl Measure for Uniform01 {}

// differential throughput measure,
//      = differential area x differential projected solid angle
//      = differential projected area x differential solid angle
//      = |W x N| * differential area * differential solid angle
#[derive(Copy, Clone, Debug)]
pub struct Throughput {}
impl Measure for Throughput {}

// TODO: define some other PDF-like structs, i.e. Spectral Radiance, Spectral Irradiance, etc

// misc traits
pub trait Abs {
    fn abs(self) -> Self;
}

impl Abs for f32 {
    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }
}

impl Abs for f32x4 {
    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }
}

pub trait One {
    const ONE: Self;
}
pub trait Zero {
    const ZERO: Self;
}

// impl One for f32 {
//     const ONE: Self = 1.0f32;
// }
// impl One for f32x4 {
//     const ONE: Self = f32x4::splat(1.0);
// }

// impl Zero for f32 {
//     const ZERO: Self = 0.0f32;
// }
// impl Zero for f32x4 {
//     const ZERO: Self = f32x4::splat(0.0);
// }

pub trait MyPartialOrd {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering>;
}

impl MyPartialOrd for f32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(self, other)
    }
}

impl MyPartialOrd for f32x4 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.gt(*other).all() {
            Some(Ordering::Greater)
        } else if self.eq(other) {
            Some(Ordering::Equal)
        } else if self.lt(*other).all() {
            Some(Ordering::Less)
        } else {
            None
        }
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum CheckResult {
    None,
    Some,
    All,
}

impl CheckResult {
    pub fn coerce(self, middle_destination: bool) -> bool {
        match self {
            CheckResult::All => true,
            CheckResult::Some => middle_destination,
            CheckResult::None => false,
        }
    }
}

pub trait CheckNAN {
    fn check_nan(&self) -> CheckResult;
}

pub trait CheckInf {
    fn check_inf(&self) -> CheckResult;
}

impl CheckNAN for f32 {
    fn check_nan(&self) -> CheckResult {
        if self.is_nan() {
            CheckResult::All
        } else {
            CheckResult::None
        }
    }
}

impl CheckNAN for f32x4 {
    fn check_nan(&self) -> CheckResult {
        let mask = self.is_nan();
        if mask.all() {
            CheckResult::All
        } else if mask.any() {
            CheckResult::Some
        } else {
            CheckResult::None
        }
    }
}

impl CheckInf for f32 {
    fn check_inf(&self) -> CheckResult {
        if self.is_infinite() {
            CheckResult::All
        } else {
            CheckResult::None
        }
    }
}

impl CheckInf for f32x4 {
    fn check_inf(&self) -> CheckResult {
        let mask = self.is_infinite();
        if mask.all() {
            CheckResult::All
        } else if mask.any() {
            CheckResult::Some
        } else {
            CheckResult::None
        }
    }
}

pub trait Field:
    Add<Output = Self>
    + AddAssign
    + Mul<Output = Self>
    + MulAssign
    + Neg<Output = Self>
    + Div<Output = Self>
    + Abs
    + Clone
    + Copy
    + PartialEq
    + MyPartialOrd
    + CheckInf
    + CheckNAN
    + Debug
{
    // trait bound to represent data types that can be integrated over.
    // examples would include f32 and f32x4
    const ZERO: Self;
    const ONE: Self;
    fn min(&self, other: Self) -> Self;
    fn max(&self, other: Self) -> Self;
}

pub trait Scalar: Field + PartialOrd {}

pub trait ToScalar<T: Field, S: Scalar> {
    fn to_scalar(v: T) -> S;
}

impl Field for f32 {
    const ONE: Self = 1.0;
    const ZERO: Self = 0.0;
    #[inline(always)]
    fn max(&self, other: Self) -> Self {
        f32::max(*self, other)
    }
    #[inline(always)]
    fn min(&self, other: Self) -> Self {
        f32::max(*self, other)
    }
}
impl Scalar for f32 {}

impl Field for f32x4 {
    const ONE: Self = f32x4::splat(1.0);
    const ZERO: Self = f32x4::splat(0.0);
    #[inline(always)]
    fn max(&self, other: Self) -> Self {
        f32x4::max(*self, other)
    }
    #[inline(always)]
    fn min(&self, other: Self) -> Self {
        f32x4::min(*self, other)
    }
}

impl ToScalar<f32x4, f32> for f32x4 {
    #[inline(always)]
    fn to_scalar(v: f32x4) -> f32 {
        v.extract(0)
    }
}
impl ToScalar<f32, f32> for f32 {
    // noop
    #[inline(always)]
    fn to_scalar(v: f32) -> f32 {
        v
    }
}
