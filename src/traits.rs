use crate::spaces::{
    Circle, DirectionalSector, DiskSpace, Element, ProductSet, R1, SimpleSet,
    SpaceParameterization, SphericalCoordinates,
};

use crate::prelude::*;
use std::f32::consts::FRAC_PI_2;
pub(crate) use std::ops::{Add, Div, Mul, Neg};
use std::{
    // arch::aarch64::float32x4x3_t,
    cmp::Ordering,
    fmt::Debug,
    marker::PhantomData,
    num::NonZero,
    ops::{AddAssign, MulAssign},
    simd::f32x2,
};

// TODO: figure out if it's necessary to create a separate trait for the support of a measure,
// i.e. R for Uniform01, or R^2 for Area, S^2 for Solid Angle, H+ for Projected Solid Angle, etc
// benefit would be allowing for other traits/structs/generics to reference the support without specifying a measure

// TODO: implement a sampling trait that allows for sampling within a specified set that is a member of the support of a measure
// i.e. sample uniformly within an interval, sample uniformly within a set of directions, or within a volume, etc

pub trait Measure {
    type Space: SpaceParameterization;
    /// measure a set
    fn measure(set: SimpleSet<Self>) -> f32;
    /// differential measure at a point. if the space/parameterization is uniform, then the differential measure will just be 1.
    /// if the space/parameterization is uniform and the measure is a pdf, then the differential measure will likely just be 1 / mu(Omega)
    /// where mu is the measure and Omega is the entire space over which the pdf is defined
    fn differential_measure(element: Element<Self>) -> f32;
}
#[derive(Copy, Clone, Debug, Default)]
pub struct ProductMeasure<A: Measure, B: Measure> {
    pub a: A,
    pub b: B,
}

impl<A: Measure, B: Measure> Measure for ProductMeasure<A, B> {
    type Space = ProductSet<A::Space, B::Space>;

    fn measure(set: SimpleSet<Self>) -> f32 {
        A::measure(set.0) * B::measure(set.1)
    }
    fn differential_measure(element: Element<Self>) -> f32 {
        A::differential_measure(element.0) * B::differential_measure(element.1)
    }
}

/* pub trait PDF: Measure {
    fn verify() -> bool {
        let space = <<Self as Measure>::Space as SpaceParameterization>::SPACE;
        (Self::measure(space) - 1.0) < 0.0001
    }
} */

/// basic lebesgue length measure
#[derive(Copy, Clone, Debug, Default)]
pub struct Length;
impl Measure for Length {
    type Space = R1;
    fn measure(set: SimpleSet<Self>) -> f32 {
        set.span()
    }
    fn differential_measure(_: Element<Self>) -> f32 {
        1.0
    }
}

/// area measure, the standard one formed by the product measure of two standard lebesgue length measures
pub type Area = ProductMeasure<Length, Length>;

/// volume measure, the standard one formed by the product measure of three standard lebesgue length measures
pub type Volume = ProductMeasure<Area, Length>;

pub struct Angle;

impl Measure for Angle {
    type Space = Circle;
    fn measure(set: SimpleSet<Self>) -> f32 {
        set.span() % Self::Space::SPACE.span()
    }
    fn differential_measure(_: Element<Self>) -> f32 {
        1.0
    }
}

pub struct DiskAreaMeasure;

impl Measure for DiskAreaMeasure {
    type Space = DiskSpace;

    fn measure(set: SimpleSet<Self>) -> f32 {
        // set.0 is angle bounds and set.1 is radius bounds

        // this formula (and the jacobian in differential_measure) can be
        // derived from the parameterization and change of variables / jacobian, then integration over the set bounds

        set.0.span() % Self::Space::SPACE.0.span() / 2.0
            * (set.1.upper.powi(2) - set.1.lower.powi(2))
    }

    fn differential_measure(element: Element<Self>) -> f32 {
        element.1
    }
}

/// solid angle measure, defined on the set of directions
/// measures the "size" of a set of 3d unit vectors / directions, where the measure of the whole sphere/set is 4pi
/// when in differential form, represents an infinitesimal increase in solid angle.
///      = sin(theta) d[theta] d[phi]
///      = d[cos theta] d[phi]
// #[derive(Copy, Clone, Debug, Default)]
pub struct SolidAngle<P: SpaceParameterization>(PhantomData<P>);
impl Measure for SolidAngle<SphericalCoordinates> {
    type Space = SphericalCoordinates;

    fn measure(set: SimpleSet<Self>) -> f32 {
        let azimuthal = set.x.span();
        let Bounds1D {
            lower: phi0,
            upper: phi1,
        } = set.y;
        // measure is the integral of the differential measure over the integration bounds
        // int_theta0^theta1 { int_phi0^phi1 { sin(phi) } }
        // == (theta1-theta0) * (-cos(phi1) + cos(phi0))
        (phi0.cos() - phi1.cos()) * azimuthal
    }

    fn differential_measure(element: Element<Self>) -> f32 {
        element.1.sin()
    }
}

impl Measure for SolidAngle<DirectionalSector> {
    type Space = DirectionalSector;
    fn measure(set: SimpleSet<Self>) -> f32 {
        TAU * (1.0 - set.1.cos())
    }
    fn differential_measure(element: Element<Self>) -> f32 {
        1.0
    }
}

/// projected solid angle measure, defined on the set of directions
/// the measure of a whole hemisphere is pi
/// represents the solid angle measure shrinked by a factor of the cosine of the angle and the surface normal
///      = |W x N| * differential solid angle
///      = |cos(theta)| sin(theta) d[theta] d[phi]
///      = |cos(theta)| d[cos theta] d[phi]
///      = sin(theta) d[sin(theta)] dphi
#[derive(Copy, Clone, Debug, Default)]
pub struct ProjectedSolidAngle {}
impl Measure for ProjectedSolidAngle {
    type Space = SphericalCoordinates;
    fn measure(set: SimpleSet<Self>) -> f32 {
        let azimuthal = set.x.span();
        let phi_bounds = set.y;
        // measure is azimuthal * int_phi0^phi1 { |cos(phi)| sin(phi) }
        // split the integrand across the phi=pi/2 boundary
        // in either case the integral of cos(phi)*sin(phi) needs to be known
        // half/double angle formula?
        // sin(2x)/2 = sin(x)*cos(x)
        // 0.25 * (cos(2*lower) - cos(2 * higher))
        // btw, cos of (2 * PI/2) == cos(pi) == -1
        // meaning the integral is either 0.25 * (-1 - cos(2phi)) if lower == pi/2, thus upper is larger than pi/2
        // or 0.25 * (cos(2phi) + 1) if upper == pi/2, thus lower is less than pi/2
        if phi_bounds.contains(&FRAC_PI_2) {
            // handle boundary crossing case
            let (phi0, phi1) = (phi_bounds.lower, phi_bounds.upper);
            0.25 * (1.0 + (2.0 * phi1).cos() + 1.0 + (2.0 * phi0).cos())
        } else {
            0.25 * ((phi_bounds.lower * 2.0).cos() - (phi_bounds.upper * 2.0).cos())
        }
    }
    fn differential_measure(element: Element<Self>) -> f32 {
        element.1.cos().abs() * element.1.sin()
    }
}

/// throughput measure, also known as the geometric measure on ray space in veach's thesis
/// measures the light-carrying capacity of a set of rays
/// in differential form, can be thought of as the
///      differential area x differential projected solid angle
///      or the differential projected area x differential solid angle
///      = |w . N| * differential area * differential solid angle
#[derive(Copy, Clone, Debug, Default)]
pub struct Throughput {}
//impl Measure for Throughput {}

/// the path throughput measure is the product measure of multiple normal Throughput measures, determined by the rank
#[derive(Debug, Copy, Clone)]
pub struct PathThroughput {
    pub rank: usize,
}

impl Default for PathThroughput {
    fn default() -> Self {
        Self { rank: 1 }
    }
}

/*impl Measure for PathThroughput {
    fn combine(self, other: Self) -> Self {
        Self {
            rank: self.rank + other.rank,
        }
    }
}

*/

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
    fn powf(self, other: Self) -> Self;
}

#[cfg(feature = "simdfloat_patch")]
impl SimdFloatPatch for f32x2 {
    fn powf(mut self, power: f32x2) -> Self {
        self[0] = self[0].powf(power[0]);
        self[1] = self[1].powf(power[1]);
        self
    }
}

#[cfg(feature = "simdfloat_patch")]
impl SimdFloatPatch for f32x4 {
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

    use num_traits::FromPrimitive;

    use super::*;

    // type DiskPDF = PDF<f32, DiskMeasure>;
    // type Sampled1D = (Sample1D, PDF<f32, Length>);
    // struct SampledDisk(pub Sample2D, pub DiskPDF);
    // impl SampledDisk {
    //     pub fn new(sample0: Sampled1D, sample1: Sampled1D) -> Self {
    //         let radial = sample0.0.x.sqrt();
    //         let angle = sample1.0.x * TAU;
    //         // jacobian matrix =
    //         /*[
    //             [ 1/(2sqrt(x)), 0],
    //             [0, TAU]
    //             jacobian determinant = PI / sqrt(x)
    //         ]*/
    //         let (sin, cos) = angle.sin_cos();
    //         // this is using Sample2D in a very nonstandard manner relative to how i've used it so far, but yeah
    //         let disk_pos = Sample2D::new(radial * cos, radial * sin);
    //         let jacobian = PI * radial.recip();
    //         Self(disk_pos, DiskPDF::new(jacobian * *sample0.1 * *sample1.1))
    //     }
    // }

    // // TODO: define some other PDF-like structs, i.e. Spectral Radiance, Spectral Irradiance, etc
    // // ideas:
    // // implement some trait called Measurable
    // // that looks something like

    // trait MonteCarlo<D: Field, M: Measure>: Field + Div<PDF<D, M>, Output = Self> {}

    // // then we can define something like

    // impl<M: Measure> Div<PDF<f32, M>> for f32 {
    //     type Output = f32;
    //     fn div(self, rhs: PDF<f32, M>) -> Self::Output {
    //         self / *rhs
    //     }
    // }
    // impl<M: Measure> MonteCarlo<f32, M> for f32 {}

    // // then if we want to measure the area under some function, we can express that integration problem using trait bounds
    // // this is somewhat generalized over what method is used to actually generate the samples

    // fn mc_integrate<DomainField, RangeField, M, F, Sampler, S>(
    //     func: F,
    //     mut sampler: Sampler,
    //     samples: u32,
    // ) -> (RangeField, RangeField)
    // where
    //     M: Measure,
    //     RangeField: MonteCarlo<RangeField, M>
    //         + Div<PDF<RangeField, M>, Output = RangeField>
    //         + Div<RangeField, Output = RangeField>
    //         + FromScalar<S>,
    //     F: Fn(DomainField) -> RangeField,
    //     Sampler: FnMut(u32) -> (DomainField, PDF<RangeField, M>),
    //     S: Scalar + FromPrimitive,
    // {
    //     let mut estimate = RangeField::ZERO;
    //     let mut sos_estimate = RangeField::ZERO;
    //     let n = RangeField::from_scalar(S::from_u32(samples).unwrap());
    //     for idx in 0..samples {
    //         let (sample, pdf) = sampler(idx);
    //         let fs = func(sample);
    //         let fpdf = fs / pdf;
    //         estimate += fpdf;
    //         sos_estimate += fpdf * fpdf;
    //     }
    //     (estimate / n, sos_estimate / n)
    // }

    // #[test]
    // fn test_mc_integral_of_a_disk() {
    //     let (estimate, square_estimate) = mc_integrate::<_, _, Area, _, _, _>(
    //         |v: Vec3| if v.x().hypot(v.y()) < 1.0 { 1.0 } else { 0.0 },
    //         |_| {
    //             let mut sample2d = Sample2D::new_random_sample();

    //             (Vec3::new(sample2d.x, sample2d.y, 0.0), PDF::new(1.0 / 4.0))
    //             // (v, PDF::new(1.0 / PI))
    //         },
    //         10000,
    //     );
    //     let variance = square_estimate - estimate * estimate;

    //     let true_value = PI;
    //     assert!(((true_value - estimate) / true_value).abs() < 0.01);
    // }

    // #[test]
    // fn test_mc_integral_of_x_cubed() {
    //     let true_value = 0.25;

    //     let (estimate, square_estimate) = mc_integrate::<_, _, Length, _, _, _>(
    //         |x: f32| x * x * x,
    //         |_| {
    //             // uniform sampling
    //             let sample = Bounds1D::new(0.0, 1.0).sample(debug_random());
    //             (sample, PDF::new(1.0))
    //         },
    //         100,
    //     );
    //     // let variance = square_estimate - estimate * estimate;
    //     // println!("{:?}, var = {:?}", estimate, variance);
    //     assert!(((estimate - true_value) / true_value).abs() < 0.2);

    //     let (estimate, square_estimate) = mc_integrate::<_, _, Length, _, _, _>(
    //         |x: f32| x * x * x,
    //         |_| {
    //             // importance sampling y=x

    //             let b = Bounds1D::new(0.0, 1.0);
    //             let c = b.span() / 2.0;

    //             let u = debug_random();
    //             let x = u.sqrt();
    //             let sample = b.sample(x);
    //             let pdf = x / c;
    //             (sample, PDF::new(pdf))
    //         },
    //         100,
    //     );
    //     let variance = square_estimate - estimate * estimate;

    //     assert!(((estimate - true_value) / true_value).abs() < 0.15);
    // }

    // #[test]
    // fn test_mc_integral_of_solid_angle() {
    //     let (estimate, square_estimate) = mc_integrate::<_, _, SolidAngle, _, _, _>(
    //         |v: Vec3| 1.0,
    //         |_| {
    //             let sample_2d = Sample2D::new_random_sample();
    //             let on_unit_sphere = random_on_unit_sphere(sample_2d);

    //             (on_unit_sphere, PDF::new(1.0 / 4.0 / PI))
    //         },
    //         10000,
    //     );
    //     let variance = square_estimate - estimate * estimate;
    //     println!("{}, stddev: {}", estimate, variance.abs().sqrt());
    // }

    // #[test]
    // fn test_mc_integral_of_projected_solid_angle() {
    //     let (estimate, square_estimate) = mc_integrate::<_, _, ProjectedSolidAngle, _, _, _>(
    //         |v: Vec3| v.z().abs(),
    //         |_| {
    //             let sample_2d = Sample2D::new_random_sample();
    //             let mut on_unit_sphere = random_on_unit_sphere(sample_2d);
    //             on_unit_sphere = Vec3::new(
    //                 on_unit_sphere.x(),
    //                 on_unit_sphere.y(),
    //                 on_unit_sphere.z().abs(),
    //             );

    //             // is distributed uniformly on half of the unit sphere, so the pdf is 1 / 2pi
    //             (on_unit_sphere, PDF::new(1.0 / 2.0 / PI))
    //         },
    //         10000000,
    //     );
    //     let variance = square_estimate - estimate * estimate;
    //     println!("{} {}", estimate, variance.abs().sqrt());

    //     let (estimate, square_estimate) = mc_integrate::<_, _, ProjectedSolidAngle, _, _, _>(
    //         |v: Vec3| v.z().abs(),
    //         |_| {
    //             let mut sample_2d = Sample2D::new_random_sample();

    //             let random_on_hemisphere = random_cosine_direction(sample_2d);
    //             (
    //                 random_on_hemisphere,
    //                 PDF::new(random_on_hemisphere.z() / PI),
    //             )
    //         },
    //         100000,
    //     );

    //     let variance = square_estimate - estimate * estimate;
    //     println!("{} {}", estimate, variance.abs().sqrt());
    // }
}
