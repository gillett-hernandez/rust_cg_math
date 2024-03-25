use crate::prelude::*;
pub(crate) use std::ops::{Add, Div, Mul, Neg};
use std::{
    cmp::Ordering,
    fmt::Debug,
    ops::{AddAssign, MulAssign}, simd::f32x2,
};

// TODO: figure out if it's necessary to create a separate trait for the support of a measure,
// i.e. R for Uniform01, or R^2 for Area, or H+ for Projected Solid Angle, etc
// differential forms of various measures
#[allow(unused_variables)]
pub trait Measure: Copy + Clone + Debug + Default + Sized {
    fn combine(self, other: Self) -> Self {
        Self::default()
    }
}

// differential solid angle
//      = sin(theta) d[theta] d[phi]
//      = d[cos theta] d[phi]
#[derive(Copy, Clone, Debug, Default)]
pub struct SolidAngle {}
impl Measure for SolidAngle {}

// differential projected solid angle
//      = |W x N| * differential solid angle
//      = |cos(theta)| sin(theta) d[theta] d[phi]
//      = |cos(theta)| d[cos theta] d[phi]
//      = sin(theta) d[sin(theta)] dphi
#[derive(Copy, Clone, Debug, Default)]
pub struct ProjectedSolidAngle {}
impl Measure for ProjectedSolidAngle {}

#[derive(Copy, Clone, Debug, Default)]
pub struct Area {}
impl Measure for Area {}

// basic measure
#[derive(Copy, Clone, Debug, Default)]
pub struct Uniform01 {}
impl Measure for Uniform01 {}

// differential throughput measure,
//      = differential area x differential projected solid angle
//      = differential projected area x differential solid angle
//      = |W x N| * differential area * differential solid angle
#[derive(Copy, Clone, Debug, Default)]
pub struct Throughput {}
impl Measure for Throughput {}

#[derive(Debug, Copy, Clone)]
pub struct PathThroughput {
    pub rank: usize,
}

impl Default for PathThroughput {
    fn default() -> Self {
        Self { rank: 1 }
    }
}

impl Measure for PathThroughput {}

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
        // disambiguation needed because this method ^ and this method v share the same name
        std::simd::num::SimdFloat::abs(self)
    }
}

pub trait TotalPartialOrd {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering>;
}

impl TotalPartialOrd for f32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(self, other)
    }
}

impl TotalPartialOrd for f32x4 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.eq(other) {
            Some(Ordering::Equal)
        } else if self.simd_ge(*other).all() {
            Some(Ordering::Greater)
        } else if self.simd_le(*other).all() {
            Some(Ordering::Less)
        } else {
            None
        }
        /* if self.gt(*other).all() {
            Some(Ordering::Greater)
        } else if self.eq(other) {
            Some(Ordering::Equal)
        } else if self.lt(*other).all() {
            Some(Ordering::Less)
        } else {
            None
        } */
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
    + TotalPartialOrd
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

// NOTE: the reason we have to implement these (ToScalar, FromScalar, CheckInf, CheckNAN, MyPartialCmp)
// as custom traits instead of using From, Into, etc is because we can't directly implement From or Into on external types

// due to rusts' rules on implementing traits
// we're allowed to implement a local trait on external types, (FromScalar for f32x4)
// or external traits on local types, (From<f32x4> for Vec3)
// but not external traits on external types (From<f32> for f32x4)

pub trait Scalar: Field + PartialOrd {}

pub trait ToScalar<S: Scalar> {
    fn to_scalar(&self) -> S;
}

pub trait FromScalar<S: Scalar> {
    fn from_scalar(v: S) -> Self;
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
    const ONE: Self = f32x4::from_array([1.0, 1.0, 1.0, 1.0]);
    const ZERO: Self = f32x4::from_array([0.0, 0.0, 0.0, 0.0]);
    #[inline(always)]
    fn max(&self, other: Self) -> Self {
        f32x4::simd_max(*self, other)
    }
    #[inline(always)]
    fn min(&self, other: Self) -> Self {
        f32x4::simd_min(*self, other)
    }
}

impl ToScalar<f32> for f32x4 {
    #[inline(always)]
    fn to_scalar(&self) -> f32 {
        self[0]
    }
}
impl ToScalar<f32> for f32 {
    // noop
    #[inline(always)]
    fn to_scalar(&self) -> f32 {
        *self
    }
}

impl FromScalar<f32> for f32x4 {
    #[inline(always)]
    fn from_scalar(v: f32) -> f32x4 {
        f32x4::splat(v)
    }
}
impl FromScalar<f32> for f32 {
    // noop
    #[inline(always)]
    fn from_scalar(v: f32) -> f32 {
        v
    }
}


#[cfg(feature = "simdfloat_patch")]
pub trait SimdFloatPatch {
    fn exp(self) -> Self;
    fn powf(self, other: Self) -> Self;
}

#[cfg(feature = "simdfloat_patch")]
impl SimdFloatPatch for f32x2 {
    fn exp(mut self) -> Self {
        self[0] = self[0].exp();
        self[1] = self[1].exp();
        self
    }
    fn powf(mut self, power: f32x2) -> Self {
        self[0] = self[0].powf(power[0]);
        self[1] = self[1].powf(power[1]);
        self
    }
}


#[cfg(feature = "simdfloat_patch")]
impl SimdFloatPatch for f32x4 {
    fn exp(mut self) -> Self {
        self[0] = self[0].exp();
        self[1] = self[1].exp();
        self[2] = self[2].exp();
        self[3] = self[3].exp();
        self
    }
    fn powf(mut self, power: f32x4) -> Self {
        self[0] = self[0].powf(power[0]);
        self[1] = self[1].powf(power[1]);
        self[2] = self[2].powf(power[2]);
        self[3] = self[3].powf(power[3]);
        self
    }
}


#[cfg(test)]
mod test {
    use std::f32::consts::TAU;

    use super::*;
    // TODO: implement trait for PDF and Measure so that you can more easily construct a new PDF on a new measure from existing pdfs, i.e.

    // subset of R^2
    #[derive(Copy, Clone, Debug, Default)]
    struct DiskMeasure {}
    impl Measure for DiskMeasure {}

    type DiskPDF = PDF<f32, DiskMeasure>;
    type Sampled1D = (Sample1D, PDF<f32, Uniform01>);
    struct SampledDisk(pub Sample2D, pub DiskPDF);
    impl SampledDisk {
        pub fn new(sample0: Sampled1D, sample1: Sampled1D) -> Self {
            let radial = sample0.0.x.sqrt();
            let angle = sample1.0.x * TAU;
            // jacobian matrix =
            /*[
                [ 1/(2sqrt(x)), 0],
                [0, TAU]
                jacobian determinant = PI / sqrt(x)
            ]*/
            let (sin, cos) = angle.sin_cos();
            // this is using Sample2D in a very nonstandard manner relative to how i've used it so far, but yeah
            let disk_pos = Sample2D::new(radial * cos, radial * sin);
            let jacobian = PI * radial.recip();
            Self(disk_pos, DiskPDF::new(jacobian * *sample0.1 * *sample1.1))
        }
    }

    // TODO: define some other PDF-like structs, i.e. Spectral Radiance, Spectral Irradiance, etc
    // ideas:
    // implement some trait called Measurable
    // that looks something like

    trait MonteCarlo<D: Field, M: Measure>: Field + Div<PDF<D, M>, Output = Self> {}

    // then we can define something like

    impl Div<PDF<f32, Uniform01>> for f32 {
        type Output = f32;
        fn div(self, rhs: PDF<f32, Uniform01>) -> Self::Output {
            self / *rhs
        }
    }
    impl MonteCarlo<f32, Uniform01> for f32 {}

    // impl Div<PDF<f32, Area>> for f32 {
    //     type Output = f32;
    //     fn div(self, rhs: PDF<f32, Area>) -> Self::Output {
    //         self / *rhs
    //     }
    // }

    // then if we want to measure the area under some function, we can express that integration problem using trait bounds
    // this is somewhat generalized over what method is used to actually generate the samples

    fn mc_integrate<DF, RF, M: Measure, F, Sampler, S>(
        func: F,
        bounds: Bounds1D,
        mut sampler: Sampler,
        samples: u16,
    ) -> (RF, RF)
    where
        DF: Field,
        RF: MonteCarlo<DF, Uniform01>
            + Div<PDF<DF, M>, Output = RF>
            + Div<RF, Output = RF>
            + FromScalar<S>,
        F: Fn(DF) -> RF,
        Sampler: FnMut(Bounds1D, u16) -> (DF, PDF<DF, M>),
        S: Scalar + From<u16>,
    {
        let mut estimate = RF::ZERO;
        let mut sos_estimate = RF::ZERO;
        let n = RF::from_scalar(samples.into());
        for idx in 0..samples {
            let (sample, pdf) = sampler(bounds, idx);
            let fs = func(sample);
            estimate += fs / pdf;
            sos_estimate += fs * fs / pdf;
        }
        (estimate / n, sos_estimate / n)
    }
    #[test]
    fn test_mc_integrate() {
        let (estimate, square_estimate) = mc_integrate(
            |x: f32| x * x * x,
            Bounds1D::new(0.0, 1.0),
            |b, _| {
                // uniform sampling
                let sample = b.sample(debug_random());
                let pdf = 1.0 / b.span();
                (sample, PDF::new(pdf))
            },
            100,
        );
        let variance = square_estimate - estimate * estimate;
        println!("{:?}, var = {:?}", estimate, variance);

        let (estimate, square_estimate) = mc_integrate(
            |x: f32| x * x * x,
            Bounds1D::new(0.0, 1.0),
            |b, _| {
                // importance sampling y=x

                let c = b.span() / 2.0;

                let u = debug_random();
                let x = u.sqrt();
                let sample = b.sample(x);
                let pdf = x / c;
                (sample, PDF::new(pdf))
            },
            100,
        );
        let variance = square_estimate - estimate * estimate;
        println!("{:?}, var = {:?}", estimate, variance);

        // let (estimate, square_estimate) = mc_integrate(
        //     |x: f32| x * x * x,
        //     Bounds1D::new(0.0, 1.0),
        //     |b, _| {
        //         // uniform sampling
        //         let sample = b.sample(debug_random());
        //         let pdf = 1.0 / b.span();
        //         (sample, PDF::new(pdf))
        //     },
        //     100,
        // );
        // let variance = square_estimate - estimate * estimate;
        // println!("{:?}, var = {:?}", estimate, variance);
    }
}
