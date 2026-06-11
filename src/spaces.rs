// use std::marker::PhantomData;
// use typenum::Unsigned;

use crate::prelude::*;

use std::f32::consts::{FRAC_PI_2, TAU};

// pub struct R<T: Unsigned>(PhantomData<T>);

/// A measurable set, independent of any chart used to coordinatize it. This is
/// the underlying *domain* a measure lives on (Veach §3.6.3: e.g. the set of
/// directions has physical meaning, but the (θ,φ) chart that parameterizes it
/// does not). A `Measure` is defined over one `Domain`; a `Parameterization`
/// charts one `Domain`; two measures over the *same* `Domain` (solid angle and
/// projected solid angle over `Directions`) can be related by a Radon–Nikodym
/// factor.
///
/// This trait is just a marker — the concrete representation of subsets and
/// points lives on the `Parameterization` (chart), not here.
pub trait Domain: Default + Clone + Copy {}

/// The real line `ℝ` (charted by [`R`] and, on a subset, [`UnitInterval`]).
#[derive(Default, Copy, Clone, Debug)]
pub struct RealLine;
impl Domain for RealLine {}

/// The set of points of the circle `S¹` (charted by [`Circle`]).
#[derive(Default, Copy, Clone, Debug)]
pub struct Angles;
impl Domain for Angles {}

/// The set of unit directions (charted by [`SphericalCoordinates`] or
/// [`DirectionalSector`]). Both [`crate::traits::SolidAngle`] and
/// [`crate::traits::ProjectedSolidAngle`] are measures over this single domain.
#[derive(Default, Copy, Clone, Debug)]
pub struct Directions;
impl Domain for Directions {}

/// Cartesian product of two domains (the domain of a [`ProductSet`] chart and a
/// `ProductMeasure`).
#[derive(Default, Copy, Clone, Debug)]
pub struct ProductDomain<A: Domain, B: Domain> {
    pub a: A,
    pub b: B,
}
impl<A: Domain, B: Domain> Domain for ProductDomain<A, B> {}

/// A chart that coordinatizes a [`Domain`]: it fixes a representation of
/// measurable subsets (`SimpleSet`) and points (`Element`) of that domain, and
/// thereby a Lebesgue reference measure on the coordinates against which a
/// measure's density (`differential_measure`) is taken. The measure's *identity*
/// lives on [`crate::traits::Measure`]; this only supplies the coordinates.
pub trait Parameterization: Default + Clone + Copy {
    /// The (chart-independent) set this chart coordinatizes.
    type Domain: Domain;
    type SimpleSet;
    type Element;
    const SPACE: Self::SimpleSet;
}

/// Former name for [`Parameterization`] (which now carries a `type Domain`).
/// Retained as an alias so existing callers — and the downstream tracer — keep
/// compiling during migration; prefer `Parameterization` in new code.
pub use self::Parameterization as SpaceParameterization;

pub type SimpleSet<P> = <P as Parameterization>::SimpleSet;
pub type Element<P> = <P as Parameterization>::Element;

#[derive(Default, Copy, Clone, Debug)]
pub struct ProductSet<A: Parameterization, B: Parameterization> {
    pub a: A,
    pub b: B,
}

impl<A: Parameterization, B: Parameterization> Parameterization for ProductSet<A, B> {
    type Domain = ProductDomain<A::Domain, B::Domain>;
    type SimpleSet = (A::SimpleSet, B::SimpleSet);
    type Element = (A::Element, B::Element);

    const SPACE: Self::SimpleSet = (A::SPACE, B::SPACE);
}

#[derive(Default, Copy, Clone, Debug)]
pub struct R;

impl Parameterization for R {
    type Domain = RealLine;
    type SimpleSet = Bounds1D;
    type Element = f32;

    const SPACE: Self::SimpleSet = Bounds1D::new(f32::NEG_INFINITY, f32::INFINITY);
}

#[derive(Default, Copy, Clone, Debug)]
pub struct UnitInterval;

impl Parameterization for UnitInterval {
    type Domain = RealLine;
    type SimpleSet = Bounds1D;
    type Element = f32;
    const SPACE: Self::SimpleSet = Bounds1D::new(0.0, 1.0);
}

#[derive(Default, Copy, Clone, Debug)]
pub struct Circle;

impl Parameterization for Circle {
    type Domain = Angles;
    type SimpleSet = Bounds1D;
    type Element = f32;

    const SPACE: Bounds1D = Bounds1D::new(0.0, TAU);
}

/// includes the interior
pub type DiskSpace = ProductSet<Circle, UnitInterval>;

/// only the surface - space of directions, solid angle, and surface area
/// theta phi parameterization, aka spherical coordinates

#[derive(Default, Copy, Clone, Debug)]
pub struct SphericalCoordinates;

impl Parameterization for SphericalCoordinates {
    type Domain = Directions;
    type SimpleSet = Bounds2D;

    type Element = (f32, f32);

    const SPACE: Self::SimpleSet = Bounds2D::new(
        Bounds1D::new(0.0, TAU),
        Bounds1D::new(-FRAC_PI_2, FRAC_PI_2),
    );
}

/// includes interior
pub type SphericalCoordinatesBall = ProductSet<SphericalCoordinates, UnitInterval>;

#[derive(Default, Copy, Clone, Debug)]
pub struct DirectionalSector;

impl Parameterization for DirectionalSector {
    type Domain = Directions;
    // Directions are represented as raw `[f32; 3]` here rather than `Vec3<S>`
    // to keep `Parameterization` non-generic over a SIMD backend. Callers
    // that want a `Vec3<S>` convert at the boundary.
    type SimpleSet = ([f32; 3], f32);
    type Element = [f32; 3];
    const SPACE: Self::SimpleSet = ([0.0, 0.0, 1.0], PI);
}
