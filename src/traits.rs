use crate::prelude::*;
use std::fmt::Debug;
pub(crate) use std::ops::{Add, Div, Mul, Neg};

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

pub trait Field:
    Add<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + Div<Output = Self>
    + Abs
    + Clone
    + Copy
    + PartialEq
    + Debug
{
    // trait bound to represent data types that can be integrated over.
    // examples would include f32 and f32x4
    const ZERO: Self;
    const ONE: Self;
}

pub trait Scalar: Field + PartialOrd {}

pub trait ToScalar<T: Field, S: Scalar> {
    fn convert(v: T) -> S;
}

impl Field for f32 {
    const ONE: Self = 1.0;
    const ZERO: Self = 0.0;
}
impl Scalar for f32 {}

impl Field for f32x4 {
    const ONE: Self = f32x4::splat(1.0);
    const ZERO: Self = f32x4::splat(0.0);
}

impl ToScalar<f32x4, f32> for f32x4 {
    #[inline(always)]
    fn convert(v: f32x4) -> f32 {
        v.extract(0)
    }
}
impl ToScalar<f32, f32> for f32 {
    // noop
    #[inline(always)]
    fn convert(v: f32) -> f32 {
        v
    }
}
