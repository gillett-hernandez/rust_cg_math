use typenum::{Sum, U1, Unsigned};

use crate::spaces::{
    Angles, Circle, DirectionalSector, DiskSpace, Directions, Domain, Element, Parameterization,
    ProductDomain, ProductSet, R, RealLine, SimpleSet, SphericalCoordinates,
};

use crate::prelude::*;
use std::{cmp::Ordering, fmt::Debug, marker::PhantomData};

// TODO: implement a sampling trait that allows for sampling within a specified set that is a member of the support of a measure
// i.e. sample uniformly within an interval, sample uniformly within a set of directions, or within a volume, etc

/// A mathematical measure, identified by the [`Domain`] (measurable set) it is
/// defined over. This trait carries *only the identity* of the measure — it is
/// the tag on `PDF<T, M>` / `Integrand<T, M>` that makes measure-correctness
/// checkable at compile time (Veach eq. 8.9). The numeric machinery (measuring a
/// set, the Lebesgue density) is chart-dependent and lives on [`ChartedMeasure`].
///
/// Two distinct measures can share a `Domain` — `SolidAngle` and
/// `ProjectedSolidAngle` are both over [`Directions`] — and are related by a
/// Radon–Nikodym factor (see `pdf::MeasureConversion`).
pub trait Measure {
    type Domain: Domain;
}

/// A [`Measure`] evaluated through a specific chart `P`. Splitting this off
/// [`Measure`] is the point of this design: the chart's Lebesgue measure is the
/// reference for `differential_measure` (the Radon–Nikodym derivative
/// `dμ/dλ_P`), so it depends on *both* the measure and the parameterization —
/// e.g. solid angle is `sinθ` in spherical coordinates but `1` over a cone
/// `DirectionalSector`. The chart must coordinatize this measure's own domain.
pub trait ChartedMeasure<P: Parameterization>: Measure
where
    P: Parameterization<Domain = <Self as Measure>::Domain>,
{
    /// measure a set
    fn measure(set: SimpleSet<P>) -> f32;
    /// differential measure at a point. if the space/parameterization is uniform, then the differential measure will just be 1.
    /// if the space/parameterization is uniform and the measure is a pdf, then the differential measure will likely just be 1 / mu(Omega)
    /// where mu is the measure and Omega is the entire space over which the pdf is defined
    fn differential_measure(element: Element<P>) -> f32;
}

#[derive(Copy, Clone, Debug, Default)]
pub struct ProductMeasure<A: Measure, B: Measure> {
    pub a: A,
    pub b: B,
}

impl<A: Measure, B: Measure> Measure for ProductMeasure<A, B> {
    type Domain = ProductDomain<A::Domain, B::Domain>;
}

impl<A, B, PA, PB> ChartedMeasure<ProductSet<PA, PB>> for ProductMeasure<A, B>
where
    PA: Parameterization<Domain = A::Domain>,
    PB: Parameterization<Domain = B::Domain>,
    A: ChartedMeasure<PA>,
    B: ChartedMeasure<PB>,
{
    #[inline(always)]
    fn measure(set: SimpleSet<ProductSet<PA, PB>>) -> f32 {
        A::measure(set.0) * B::measure(set.1)
    }
    #[inline(always)]
    fn differential_measure(element: Element<ProductSet<PA, PB>>) -> f32 {
        A::differential_measure(element.0) * B::differential_measure(element.1)
    }
}

/// basic lebesgue length measure
#[derive(Copy, Clone, Debug, Default)]
pub struct Length;
impl Measure for Length {
    type Domain = RealLine;
}
impl ChartedMeasure<R> for Length {
    #[inline(always)]
    fn measure(set: SimpleSet<R>) -> f32 {
        set.span()
    }
    #[inline(always)]
    fn differential_measure(_: Element<R>) -> f32 {
        1.0
    }
}

/// Spectral measure `dλ` over wavelength. Physically a length (nm), but a
/// **distinct measure type** from [`Length`]: spectral selection is its own
/// integration axis (the `Λ` axis, TODO #20/#23). Keeping it distinct turns the
/// #20 bug — a wavelength pdf silently typed `PDF<_, Length>` and then dropped —
/// into a compile error, because `PDF<_, Length> ≠ PDF<_, Wavelength>` and an
/// `Integrand<_, Wavelength>` may only be divided by a `PDF<_, Wavelength>`.
///
/// Integrating over the sampled wavelength is the *ordinary* estimator division
/// (`Integrand<_, Wavelength> / PDF<_, Wavelength> → Estimate`); the measure
/// cancels, so no bespoke Radon–Nikodym conversion is needed.
///
/// ```
/// use math::prelude::*;
/// let f: Integrand<f32, Wavelength> = Integrand::new(6.0);
/// let p: PDF<f32, Wavelength> = PDF::new(2.0);
/// let est: Estimate<f32> = f / p; // integrate out λ
/// assert_eq!(*est, 3.0);
/// ```
///
/// A wavelength density cannot stand in for a length density (the #20 bug):
/// ```compile_fail
/// use math::prelude::*;
/// let f: Integrand<f32, Wavelength> = Integrand::new(1.0);
/// let p: PDF<f32, Length> = PDF::new(1.0);
/// let _ = f / p; // ERROR: no `Div` impl — Wavelength ≠ Length
/// ```
#[derive(Copy, Clone, Debug, Default)]
pub struct Wavelength;
impl Measure for Wavelength {
    type Domain = RealLine;
}
impl ChartedMeasure<R> for Wavelength {
    #[inline(always)]
    fn measure(set: SimpleSet<R>) -> f32 {
        set.span()
    }
    #[inline(always)]
    fn differential_measure(_: Element<R>) -> f32 {
        1.0
    }
}

/// area measure, the standard one formed by the product measure of two standard lebesgue length measures
pub type Area = ProductMeasure<Length, Length>;

/// volume measure, the standard one formed by the product measure of three standard lebesgue length measures
pub type Volume = ProductMeasure<Area, Length>;

pub struct Angle;

impl Measure for Angle {
    type Domain = Angles;
}
impl ChartedMeasure<Circle> for Angle {
    #[inline(always)]
    fn measure(set: SimpleSet<Circle>) -> f32 {
        set.span() % Circle::SPACE.span()
    }
    #[inline(always)]
    fn differential_measure(_: Element<Circle>) -> f32 {
        1.0
    }
}

pub struct DiskAreaMeasure;

// The disk-area measure is charted by `DiskSpace` (a product of an angle chart
// and a radius chart), so its domain is the corresponding product domain. Note
// it is NOT a `ProductMeasure`: the radius Jacobian in `differential_measure`
// couples the factors.
impl Measure for DiskAreaMeasure {
    type Domain = ProductDomain<Angles, RealLine>;
}
impl ChartedMeasure<DiskSpace> for DiskAreaMeasure {
    #[inline(always)]
    fn measure(set: SimpleSet<DiskSpace>) -> f32 {
        // set.0 is angle bounds and set.1 is radius bounds

        // this formula (and the jacobian in differential_measure) can be
        // derived from the parameterization and change of variables / jacobian, then integration over the set bounds

        set.0.span() % DiskSpace::SPACE.0.span() / 2.0
            * (set.1.upper.powi(2) - set.1.lower.powi(2))
    }

    #[inline(always)]
    fn differential_measure(element: Element<DiskSpace>) -> f32 {
        element.1
    }
}

/// solid angle measure, defined on the set of directions
/// measures the "size" of a set of 3d unit vectors / directions, where the measure of the whole sphere/set is 4pi
/// when in differential form, represents an infinitesimal increase in solid angle.
///      = sin(theta) d[theta] d[phi]
///      = d[cos theta] d[phi]
/// solid angle measure, defined on the set of directions ([`Directions`]),
/// independent of the chart used to evaluate it. The measure of the whole
/// sphere is 4π. Charted by [`SphericalCoordinates`] (`dσ = sinθ dθ dφ`) and by
/// [`DirectionalSector`] (a cone). Collapsing the former `SolidAngle<P>` into a
/// single type lets a `PDF<_, SolidAngle>` match regardless of which chart
/// produced it (Veach §3.6.3).
#[derive(Copy, Clone, Debug, Default)]
pub struct SolidAngle;
impl Measure for SolidAngle {
    type Domain = Directions;
}
impl ChartedMeasure<SphericalCoordinates> for SolidAngle {
    #[inline(always)]
    fn measure(set: SimpleSet<SphericalCoordinates>) -> f32 {
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

    #[inline(always)]
    fn differential_measure(element: Element<SphericalCoordinates>) -> f32 {
        element.1.sin()
    }
}

impl ChartedMeasure<DirectionalSector> for SolidAngle {
    #[inline(always)]
    fn measure(set: SimpleSet<DirectionalSector>) -> f32 {
        TAU * (1.0 - set.1.cos())
    }
    #[inline(always)]
    fn differential_measure(_: Element<DirectionalSector>) -> f32 {
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
    type Domain = Directions;
}
impl ChartedMeasure<SphericalCoordinates> for ProjectedSolidAngle {
    #[inline(always)]
    fn measure(set: SimpleSet<SphericalCoordinates>) -> f32 {
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
            0.25 * azimuthal * (1.0 + (2.0 * phi1).cos() + 1.0 + (2.0 * phi0).cos())
        } else {
            0.25 * azimuthal * ((phi_bounds.lower * 2.0).cos() - (phi_bounds.upper * 2.0).cos())
        }
    }
    #[inline(always)]
    fn differential_measure(element: Element<SphericalCoordinates>) -> f32 {
        element.1.cos().abs() * element.1.sin()
    }
}

/// throughput measure, also known as the geometric measure on ray space in veach's thesis
/// measures the light-carrying capacity of a set of rays
/// in differential form, can be thought of as the
///      differential area x differential projected solid angle
///      or the differential projected area x differential solid angle
///      = |w . N| * differential area * differential solid angle
pub type ThroughputMeasure = ProductMeasure<Area, ProjectedSolidAngle>;

/// Per-wavelength (spectral) throughput measure — the ray-space throughput
/// measure tensored with the spectral measure, `ThroughputMeasure × dλ`. A
/// spectral (per-λ) radiance is an `Integrand<_, SpectralThroughputMeasure>`;
/// integrating over the sampled wavelength divides by the `PDF<_, Wavelength>`
/// factor (the [`Wavelength`] measure cancels), leaving an
/// `Integrand<_, ThroughputMeasure>`.
pub type SpectralThroughputMeasure = ProductMeasure<ThroughputMeasure, Wavelength>;

/// the path throughput measure is the product measure of multiple normal ThroughputMeasure measures, determined by the rank
#[derive(Debug, Copy, Clone)]
pub struct PathThroughput<N: Unsigned>(PhantomData<N>);

impl<N: Unsigned> Default for PathThroughput<N> {
    #[inline(always)]
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<N> Mul<ThroughputMeasure> for PathThroughput<N>
where
    N: Unsigned + Add<U1>,
    Sum<N, U1>: Unsigned,
{
    type Output = PathThroughput<Sum<N, U1>>;
    #[inline(always)]
    fn mul(self, _: ThroughputMeasure) -> Self::Output {
        Self::Output::default()
    }
}

/// Concatenating two sub-paths multiplies their throughput densities, so the
/// ranks add: `PathThroughput<M> * PathThroughput<N> = PathThroughput<M+N>`.
impl<M, N> Mul<PathThroughput<N>> for PathThroughput<M>
where
    M: Unsigned + Add<N>,
    N: Unsigned,
    Sum<M, N>: Unsigned,
{
    type Output = PathThroughput<Sum<M, N>>;
    #[inline(always)]
    fn mul(self, _: PathThroughput<N>) -> Self::Output {
        Self::Output::default()
    }
}

/// Phantom [`Domain`] of the throughput measure on a path of `N` vertices: the
/// `N`-fold product of the per-bounce throughput domain. Indexed by the typenum
/// rank so paths of different length are distinct domains. Mirrors
/// [`AreaProductDomain`]; `N` is the only thing distinguishing ranks, so a phantom
/// captures the identity exactly.
pub struct PathThroughputDomain<N: Unsigned>(PhantomData<N>);
impl<N: Unsigned> Default for PathThroughputDomain<N> {
    #[inline(always)]
    fn default() -> Self {
        Self(PhantomData)
    }
}
impl<N: Unsigned> Clone for PathThroughputDomain<N> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<N: Unsigned> Copy for PathThroughputDomain<N> {}
impl<N: Unsigned> Domain for PathThroughputDomain<N> {}

/// `PathThroughput<N>` is a pure identity [`Measure`] tag (like [`AreaProduct`]):
/// it carries no concrete chart, so it has no [`ChartedMeasure`] impl, but it can
/// tag a `PDF` / `Integrand` so #1's `Integrand / PDF` division cancels two
/// throughput-measure quantities only when their vertex counts match.
impl<N: Unsigned> Measure for PathThroughput<N> {
    type Domain = PathThroughputDomain<N>;
}

/// Phantom [`Domain`] of the area-product measure on a path of `N` vertices: the
/// `N`-fold product of the surface-area domain (`ℝ²`). Indexed by the typenum
/// rank so paths of different length are distinct domains.
pub struct AreaProductDomain<N: Unsigned>(PhantomData<N>);
impl<N: Unsigned> Default for AreaProductDomain<N> {
    #[inline(always)]
    fn default() -> Self {
        Self(PhantomData)
    }
}
impl<N: Unsigned> Clone for AreaProductDomain<N> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<N: Unsigned> Copy for AreaProductDomain<N> {}
impl<N: Unsigned> Domain for AreaProductDomain<N> {}

/// The **area-product measure** on path space (Veach §8.A, `μ^a`): the product of
/// `N` copies of the surface-[`Area`] measure, one per path vertex. Path
/// contributions in a vertex-area formulation are densities w.r.t. this measure,
/// which is *distinct* from the throughput measure [`ThroughputMeasure`] /
/// [`PathThroughput`] (the latter folds in the geometry/cosine terms). Tagging a
/// `PDF` / `Integrand` with `AreaProduct<N>` lets #1's `Integrand / PDF` division
/// cancel two path quantities only when their vertex counts match.
///
/// `N` is a typenum rank, so a path can be grown in the type system:
/// `AreaProduct<N> * Area = AreaProduct<N+1>` appends one vertex (see the [`Mul`]
/// impl).
///
/// ```
/// use math::prelude::*;
/// use typenum::U3;
/// // a 3-vertex path contribution divided by its 3-vertex area-product pdf
/// let f: Integrand<f32, AreaProduct<U3>> = Integrand::new(6.0);
/// let p: PDF<f32, AreaProduct<U3>> = PDF::new(2.0);
/// let est: Estimate<f32> = f / p; // ranks match → OK
/// assert_eq!(*est, 3.0);
/// ```
///
/// A 2-vertex pdf cannot cancel a 3-vertex integrand:
///
/// ```compile_fail
/// use math::prelude::*;
/// use typenum::{U2, U3};
/// let f: Integrand<f32, AreaProduct<U3>> = Integrand::new(6.0);
/// let p: PDF<f32, AreaProduct<U2>> = PDF::new(2.0);
/// let _est = f / p; // ERROR: AreaProduct<U3> ≠ AreaProduct<U2>
/// ```
#[derive(Debug, Copy, Clone)]
pub struct AreaProduct<N: Unsigned>(PhantomData<N>);

impl<N: Unsigned> Default for AreaProduct<N> {
    #[inline(always)]
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<N: Unsigned> Measure for AreaProduct<N> {
    type Domain = AreaProductDomain<N>;
}

/// Append a vertex to a path: `AreaProduct<N> * Area = AreaProduct<N+1>`.
impl<N> Mul<Area> for AreaProduct<N>
where
    N: Unsigned + Add<U1>,
    Sum<N, U1>: Unsigned,
{
    type Output = AreaProduct<Sum<N, U1>>;
    #[inline(always)]
    fn mul(self, _: Area) -> Self::Output {
        Self::Output::default()
    }
}

/// Concatenating two sub-paths multiplies their area-product densities, so the
/// ranks add: `AreaProduct<M> * AreaProduct<N> = AreaProduct<M+N>`.
impl<M, N> Mul<AreaProduct<N>> for AreaProduct<M>
where
    M: Unsigned + Add<N>,
    N: Unsigned,
    Sum<M, N>: Unsigned,
{
    type Output = AreaProduct<Sum<M, N>>;
    #[inline(always)]
    fn mul(self, _: AreaProduct<N>) -> Self::Output {
        Self::Output::default()
    }
}

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

pub trait TotalPartialOrd {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering>;
}

impl TotalPartialOrd for f32 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(self, other)
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum CheckResult {
    None,
    Some,
    All,
}

impl CheckResult {
    #[inline(always)]
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
    #[inline(always)]
    fn check_nan(&self) -> CheckResult {
        if self.is_nan() {
            CheckResult::All
        } else {
            CheckResult::None
        }
    }
}

impl CheckInf for f32 {
    #[inline(always)]
    fn check_inf(&self) -> CheckResult {
        if self.is_infinite() {
            CheckResult::All
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
        f32::min(*self, other)
    }
}
impl Scalar for f32 {}

impl ToScalar<f32> for f32 {
    // noop
    #[inline(always)]
    fn to_scalar(&self) -> f32 {
        *self
    }
}

impl FromScalar<f32> for f32 {
    // noop
    #[inline(always)]
    fn from_scalar(v: f32) -> f32 {
        v
    }
}

// ===========================================================================
// Thermite bridge: blanket impls of our local traits over thermite's vector
// traits, so any backend-chosen thermite vector type satisfies `Field` (and
// thereby plugs into `WavelengthEnergy`, `PDF`, `Curve`, etc.) without us
// having to write per-type impls.
//
// `HeroWavelength` is `WavelengthEnergy<Vector<R>, Vector<R>>`, so these
// `Vector<R>` impls are the only spectral-vector path; the old concrete
// `std::simd::f32x4` impls were dead once that alias migrated and have been
// removed (the crate no longer needs `#![feature(portable_simd)]`).
// ===========================================================================

// The blanket impls are written over `Vector<R>` (concrete type constructor) +
// the relevant `*Register` bound, *not* over `V: FloatVector`. Coherence
// otherwise rejects them — Rust can't prove `f32` will never grow a
// `FloatVector` impl in a future thermite version, so `impl<V: FloatVector> X
// for V` is treated as potentially overlapping with `impl X for f32`. Bounding
// on `Vector<R>` is decidable: `f32` cannot be `Vector<R>` for any R.
impl<R: thermite::register::SignedRegister> Abs for Vector<R> {
    #[inline(always)]
    fn abs(self) -> Self {
        <Self as SignedVector>::abs(self)
    }
}

impl<R: thermite::register::FloatRegister> CheckNAN for Vector<R> {
    #[inline(always)]
    fn check_nan(&self) -> CheckResult {
        let mask = <Self as FloatVector>::is_nan(*self);
        if mask.all() {
            CheckResult::All
        } else if mask.any() {
            CheckResult::Some
        } else {
            CheckResult::None
        }
    }
}

impl<R: thermite::register::FloatRegister> CheckInf for Vector<R> {
    #[inline(always)]
    fn check_inf(&self) -> CheckResult {
        let mask = <Self as FloatVector>::is_infinite(*self);
        if mask.all() {
            CheckResult::All
        } else if mask.any() {
            CheckResult::Some
        } else {
            CheckResult::None
        }
    }
}

impl<R: thermite::register::FloatRegister> TotalPartialOrd for Vector<R> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if *self == *other {
            Some(Ordering::Equal)
        } else if <Self as PartialOrdVector>::cmp_ge(*self, *other).all() {
            Some(Ordering::Greater)
        } else if <Self as PartialOrdVector>::cmp_le(*self, *other).all() {
            Some(Ordering::Less)
        } else {
            None
        }
    }
}

// Lets spectral pdfs (`HeroWavelength`, i.e. `PDF<Vector<R>, _>`) flow through
// the scalar-Jacobian measure conversions in `pdf.rs`.
impl<R: thermite::register::FloatRegister<Element = f32>> FromScalar<f32> for Vector<R> {
    #[inline(always)]
    fn from_scalar(v: f32) -> Self {
        Vector::<R>::splat(v)
    }
}

impl<R: thermite::register::FloatRegister> Field for Vector<R> {
    const ZERO: Self = <Vector<R> as NumericVector>::ZERO;
    const ONE: Self = <Vector<R> as NumericVector>::ONE;
    #[inline(always)]
    fn min(&self, other: Self) -> Self {
        <Self as NumericVector>::min(*self, other)
    }
    #[inline(always)]
    fn max(&self, other: Self) -> Self {
        <Self as NumericVector>::max(*self, other)
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn area_product_extends_rank() {
        use typenum::{U2, U3};
        // appending a vertex increments the typenum rank — checked at compile time
        // by the type ascription on the binding.
        let _: AreaProduct<U3> = AreaProduct::<U2>::default() * Area::default();
    }

    #[test]
    fn path_throughput_extends_rank() {
        use typenum::{U2, U3};
        // multiplying by one ThroughputMeasure factor appends one path vertex: the rank
        // increments by U1 (not the old Add<N> doubling). Checked at compile time
        // by the type ascription on the binding.
        let _: PathThroughput<U3> =
            PathThroughput::<U2>::default() * ThroughputMeasure::default();
    }

    #[test]
    fn area_product_concatenation_adds_ranks() {
        use typenum::{U2, U3, U5};
        // joining two sub-paths multiplies their densities → ranks add.
        let _: AreaProduct<U5> = AreaProduct::<U2>::default() * AreaProduct::<U3>::default();
    }

    #[test]
    fn path_throughput_concatenation_adds_ranks() {
        use typenum::{U2, U3, U5};
        let _: PathThroughput<U5> =
            PathThroughput::<U2>::default() * PathThroughput::<U3>::default();
    }

    #[test]
    fn path_throughput_estimator_cancels() {
        use typenum::U4;
        // PathThroughput<N> is now a Measure, so #1's Integrand / PDF division
        // cancels matching path ranks into a measure-free Estimate.
        let f: Integrand<f32, PathThroughput<U4>> = Integrand::new(9.0);
        let p: PDF<f32, PathThroughput<U4>> = PDF::new(3.0);
        let est: Estimate<f32> = f / p;
        assert_eq!(*est, 3.0);
    }

    #[test]
    fn area_product_estimator_cancels() {
        use typenum::U4;
        // #1's Integrand / PDF division cancels matching path ranks, yielding a
        // measure-free Estimate.
        let f: Integrand<f32, AreaProduct<U4>> = Integrand::new(8.0);
        let p: PDF<f32, AreaProduct<U4>> = PDF::new(2.0);
        let est: Estimate<f32> = f / p;
        assert_eq!(*est, 4.0);
    }

    #[test]
    fn solidangle_measure() {
        type TestS = thermite::backend::scalar::Scalar;
        let e: Vec3<TestS> = Vec3::new(1.0, 1.0, 1.0).normalized();
        let d_mu = <SolidAngle as ChartedMeasure<DirectionalSector>>::differential_measure(
            e.as_array()[..3].try_into().unwrap(),
        );
        println!("d_mu is {}", d_mu);

        let uv = direction_to_uv(e);
        let d_mu = <SolidAngle as ChartedMeasure<SphericalCoordinates>>::differential_measure((
            uv.0 * TAU,
            uv.1 * PI,
        ));
        println!("d_mu is {}", d_mu);
    }

    // --- ChartedMeasure::measure / differential_measure closed forms ---

    #[test]
    fn length_measure_is_span() {
        assert_eq!(
            <Length as ChartedMeasure<R>>::measure(Bounds1D::new(2.0, 5.0)),
            3.0
        );
        assert_eq!(<Length as ChartedMeasure<R>>::differential_measure(0.5), 1.0);
    }

    #[test]
    fn angle_measure_wraps_circle() {
        assert!(
            (<Angle as ChartedMeasure<Circle>>::measure(Bounds1D::new(0.0, PI)) - PI).abs() < 1e-5
        );
        assert_eq!(
            <Angle as ChartedMeasure<Circle>>::differential_measure(1.0),
            1.0
        );
    }

    #[test]
    fn disk_area_measure_and_jacobian() {
        // (angle span PI) / 2 * (1² - 0²) = PI/2
        let set = (Bounds1D::new(0.0, PI), Bounds1D::new(0.0, 1.0));
        let m = <DiskAreaMeasure as ChartedMeasure<DiskSpace>>::measure(set);
        assert!((m - PI / 2.0).abs() < 1e-5, "disk area measure {}", m);
        // differential measure is the radius (the change-of-variables Jacobian)
        assert_eq!(
            <DiskAreaMeasure as ChartedMeasure<DiskSpace>>::differential_measure((0.3, 0.7)),
            0.7
        );
    }

    #[test]
    fn product_area_measure_multiplies_factors() {
        let set = (Bounds1D::new(0.0, 2.0), Bounds1D::new(0.0, 3.0));
        let m = <Area as ChartedMeasure<ProductSet<R, R>>>::measure(set);
        assert!((m - 6.0).abs() < 1e-5, "area {}", m);
        assert_eq!(
            <Area as ChartedMeasure<ProductSet<R, R>>>::differential_measure((0.5, 0.5)),
            1.0
        );
    }

    #[test]
    fn solid_angle_spherical_full_sphere_is_4pi() {
        let full = Bounds2D::new(Bounds1D::new(0.0, TAU), Bounds1D::new(0.0, PI));
        let m = <SolidAngle as ChartedMeasure<SphericalCoordinates>>::measure(full);
        assert!((m - 4.0 * PI).abs() < 1e-4, "full sphere solid angle {}", m);
        assert!(
            (<SolidAngle as ChartedMeasure<SphericalCoordinates>>::differential_measure((
                0.0,
                FRAC_PI_2
            )) - 1.0)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn solid_angle_sector_hemisphere_is_tau() {
        // cone with half-angle π/2 is a hemisphere: TAU*(1 - cos(π/2)) = TAU.
        let set = ([0.0, 0.0, 1.0], FRAC_PI_2);
        let m = <SolidAngle as ChartedMeasure<DirectionalSector>>::measure(set);
        assert!((m - TAU).abs() < 1e-5, "hemisphere sector {}", m);
        assert_eq!(
            <SolidAngle as ChartedMeasure<DirectionalSector>>::differential_measure([0.0, 0.0, 1.0]),
            1.0
        );
    }

    #[test]
    fn projected_solid_angle_both_phi_branches() {
        // Branch A: phi bounds entirely below π/2.
        let below = Bounds2D::new(Bounds1D::new(0.0, TAU), Bounds1D::new(0.0, FRAC_PI_2 * 0.5));
        let a = <ProjectedSolidAngle as ChartedMeasure<SphericalCoordinates>>::measure(below);
        assert!(a.is_finite() && a > 0.0, "below-branch measure {}", a);

        // Branch B: phi bounds straddle π/2 (contains FRAC_PI_2).
        let straddle = Bounds2D::new(Bounds1D::new(0.0, TAU), Bounds1D::new(0.0, PI));
        let b = <ProjectedSolidAngle as ChartedMeasure<SphericalCoordinates>>::measure(straddle);
        assert!(b.is_finite() && b > 0.0, "straddle-branch measure {}", b);

        // differential: |cosφ|·sinφ
        let phi = std::f32::consts::FRAC_PI_3;
        let d = <ProjectedSolidAngle as ChartedMeasure<SphericalCoordinates>>::differential_measure(
            (0.0, phi),
        );
        let expect = phi.cos().abs() * phi.sin();
        assert!((d - expect).abs() < 1e-6, "differential {} vs {}", d, expect);
    }

    // --- Vector<R> blanket helper-trait impls ---

    type TestR = <thermite::backend::scalar::Scalar as thermite::simd::Simd>::f32x4;

    #[test]
    fn vector_abs() {
        let v = Vector::<TestR>::new([-1.0, 2.0, -3.0, 4.0]);
        assert_eq!(&Abs::abs(v).into_array()[..], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn vector_check_nan_and_inf() {
        let clean = Vector::<TestR>::new([1.0, 2.0, 3.0, 4.0]);
        let some_nan = Vector::<TestR>::new([f32::NAN, 2.0, 3.0, 4.0]);
        let all_nan = Vector::<TestR>::splat(f32::NAN);
        assert_eq!(clean.check_nan(), CheckResult::None);
        assert_eq!(some_nan.check_nan(), CheckResult::Some);
        assert_eq!(all_nan.check_nan(), CheckResult::All);

        let some_inf = Vector::<TestR>::new([f32::INFINITY, 2.0, 3.0, 4.0]);
        let all_inf = Vector::<TestR>::splat(f32::INFINITY);
        assert_eq!(clean.check_inf(), CheckResult::None);
        assert_eq!(some_inf.check_inf(), CheckResult::Some);
        assert_eq!(all_inf.check_inf(), CheckResult::All);
    }

    #[test]
    fn vector_total_partial_ord() {
        let a = Vector::<TestR>::splat(1.0);
        let b = Vector::<TestR>::splat(2.0);
        assert_eq!(TotalPartialOrd::partial_cmp(&a, &a), Some(Ordering::Equal));
        assert_eq!(TotalPartialOrd::partial_cmp(&b, &a), Some(Ordering::Greater));
        assert_eq!(TotalPartialOrd::partial_cmp(&a, &b), Some(Ordering::Less));
        // incomparable: some lanes greater, some lesser
        let c = Vector::<TestR>::new([1.0, 5.0, 1.0, 5.0]);
        let d = Vector::<TestR>::new([5.0, 1.0, 5.0, 1.0]);
        assert_eq!(TotalPartialOrd::partial_cmp(&c, &d), None);
    }

    #[test]
    fn vector_from_scalar_and_field() {
        let v = <Vector<TestR> as FromScalar<f32>>::from_scalar(5.0);
        assert_eq!(&v.into_array()[..], &[5.0, 5.0, 5.0, 5.0]);

        let a = Vector::<TestR>::new([1.0, 4.0, 1.0, 4.0]);
        let b = Vector::<TestR>::new([3.0, 2.0, 3.0, 2.0]);
        assert_eq!(&Field::min(&a, b).into_array()[..], &[1.0, 2.0, 1.0, 2.0]);
        assert_eq!(&Field::max(&a, b).into_array()[..], &[3.0, 4.0, 3.0, 4.0]);
        assert_eq!(&<Vector<TestR> as Field>::ZERO.into_array()[..], &[0.0; 4]);
        assert_eq!(&<Vector<TestR> as Field>::ONE.into_array()[..], &[1.0; 4]);
    }

    #[test]
    fn f32_field_and_checks() {
        assert_eq!(Field::min(&2.0f32, 5.0), 2.0);
        assert_eq!(Field::max(&2.0f32, 5.0), 5.0);
        assert_eq!(<f32 as Field>::ZERO, 0.0);
        assert_eq!(<f32 as Field>::ONE, 1.0);
        assert_eq!(f32::NAN.check_nan(), CheckResult::All);
        assert_eq!(1.0f32.check_nan(), CheckResult::None);
        assert_eq!(f32::INFINITY.check_inf(), CheckResult::All);
        assert_eq!(1.0f32.check_inf(), CheckResult::None);
        assert_eq!(Abs::abs(-3.0f32), 3.0);
    }

    #[test]
    fn f32_total_partial_ord() {
        assert_eq!(TotalPartialOrd::partial_cmp(&1.0f32, &2.0), Some(Ordering::Less));
        assert_eq!(TotalPartialOrd::partial_cmp(&2.0f32, &2.0), Some(Ordering::Equal));
        assert_eq!(TotalPartialOrd::partial_cmp(&3.0f32, &2.0), Some(Ordering::Greater));
        assert_eq!(TotalPartialOrd::partial_cmp(&f32::NAN, &2.0), None);
    }

    #[test]
    fn phantom_domains_default_and_clone() {
        use typenum::U3;
        // exercise Default + Clone/Copy on the phantom path-domain markers
        let a = AreaProductDomain::<U3>::default();
        let _a2 = a.clone();
        let _a3 = a; // Copy
        let p = PathThroughputDomain::<U3>::default();
        let _p2 = p.clone();
        let _p3 = p; // Copy
    }

    #[test]
    fn check_result_coerce() {
        assert!(CheckResult::All.coerce(false));
        assert!(!CheckResult::None.coerce(true));
        assert!(CheckResult::Some.coerce(true));
        assert!(!CheckResult::Some.coerce(false));
    }

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
