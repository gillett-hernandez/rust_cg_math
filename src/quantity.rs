//! Radiometric-quantity newtypes — the *quantity* axis, orthogonal to the
//! *measure* axis carried by [`PDF`] / [`Integrand`] / [`Estimate`].
//!
//! The measure layer answers "a density/integrand *with respect to which*
//! measure" (Veach §3.B.5: a radiometric quantity is a ratio of measures). This
//! module answers the orthogonal question "*what physical quantity* is this
//! number" — radiance, importance, a BSDF value, or a dimensionless path
//! throughput. Mixing them up is the bug class behind non-symmetric scattering
//! (the radiance-vs-importance / adjoint distinction, Veach §3.7.3, §3.7.6,
//! §5.2.3); tagging them makes the mistake a compile error.
//!
//! This completes the dormant sketch in `traits.rs` ("define some other PDF-like
//! structs, i.e. Spectral Radiance, Spectral Irradiance … a trait called
//! `Measurable`").
//!
//! These wrap the energy field `E: Field`, so they are backend-generic: `E` can
//! be `f32` or a thermite SIMD register `Vector<R>`. Scalars that enter the
//! algebra (cosines, MIS weights) go through `FromScalar<f32>` rather than
//! hard-coded `f32` arithmetic, so the spectral/SIMD path keeps working.
//!
//! They are deliberately **not** [`Field`]: a `Field` would admit nonsensical
//! operations like `BSDF / BSDF`. Instead each carries a small, curated set of
//! `Mul` / `Div` / `Add` impls encoding only the physically meaningful algebra:
//!
//! - `Throughput × Throughput = Throughput`             (extend a path)
//! - `Throughput × Radiance   = Radiance`               (and `Importance`, `Emission`)
//! - `Radiance + Radiance`                              (accumulate; ditto `Importance`)
//! - [`BSDF::estimator`] — the bridge to the measure layer: `f·cos/pdf`
//! - `Importance × Radiance = Estimate`                 (the measurement, Veach §3.7.1)

use crate::prelude::*;
use std::ops::Deref;

/// Marker for a radiometric quantity (the `Measurable` idea from the original
/// `traits.rs` sketch). Lets generic code bound on "is a radiometric quantity"
/// without enumerating each newtype.
pub trait Quantity: Copy {
    /// The underlying energy field type (`f32`, `Vector<R>`, …).
    type Field: Field;
    /// Read the raw value out of the quantity.
    fn value(self) -> Self::Field;
}

/// Transport **role** of a radiometric quantity — which adjoint solution it
/// belongs to (Veach §3.7.3). Role is deliberately *not* a dimension: importance
/// `W_e` and radiance `L` carry the *same* dimensions, so the radiance/importance
/// distinction can only be a separate zero-cost phantom. It lets the measurement
/// bridge `⟨W_e, L⟩ = Adjoint × Primal → Estimate` (Veach §3.7.1) be typed while a
/// nonsensical `L · L` is rejected. Consumed by the Slice 3 carrier (TODO #23).
pub trait Role: Copy + Default {
    /// The opposite transport role (`Primal ↔ Adjoint`) — the partner a
    /// measurement pairs this role with.
    type Dual: Role;
}

/// The **primal** transport solution: radiance / the measurement carried toward
/// the sensor.
#[derive(Copy, Clone, Default, PartialEq, Eq, Debug)]
pub struct Primal;
/// The **adjoint** transport solution: importance carried from the sensor
/// (particle / light tracing).
#[derive(Copy, Clone, Default, PartialEq, Eq, Debug)]
pub struct Adjoint;

impl Role for Primal {
    type Dual = Adjoint;
}
impl Role for Adjoint {
    type Dual = Primal;
}

/// Generates a `pub struct Name<E>(pub E)` newtype with `Deref<Target = E>`,
/// `Quantity`, and the usual derives, so the per-quantity blocks below only have
/// to spell out the algebra that is unique to them.
macro_rules! quantity {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
        pub struct $name<E: Field>(pub E);

        impl<E: Field> Deref for $name<E> {
            type Target = E;
            #[inline(always)]
            fn deref(&self) -> &E {
                &self.0
            }
        }

        impl<E: Field> Quantity for $name<E> {
            type Field = E;
            #[inline(always)]
            fn value(self) -> E {
                self.0
            }
        }
    };
}

quantity!(
    /// Radiance `L` (W·m⁻²·sr⁻¹), the quantity transported toward the camera
    /// (Veach §3.4.3). What a path tracer accumulates.
    Radiance
);
quantity!(
    /// Importance `W_e`, the adjoint of radiance — the quantity transported from
    /// the sensor in particle/light tracing (Veach §3.7.3).
    Importance
);
quantity!(
    /// Emitted exitant radiance `L_e` (Veach §3.5). Enters transport as
    /// [`Radiance`] (see the `From` impl).
    Emission
);
quantity!(
    /// Irradiance `E` (W·m⁻²) — radiance integrated over the projected hemisphere
    /// (Veach §3.4.2).
    Irradiance
);
quantity!(
    /// A bidirectional scattering distribution value `f_s` (units sr⁻¹),
    /// Veach §3.6. Combine with a cosine and a directional density via
    /// [`BSDF::estimator`] to get a dimensionless [`Throughput`] factor.
    BSDF
);
quantity!(
    /// Dimensionless path throughput `β`: the running product of `f·cos/pdf`
    /// ratios along a path. Carries no units — it scales a transported quantity.
    Throughput
);

// ---------------------------------------------------------------------------
// Throughput: multiplicative, dimensionless.
// ---------------------------------------------------------------------------

impl<E: Field> Throughput<E> {
    /// The multiplicative identity throughput (a fresh path carries `β = 1`).
    #[inline(always)]
    pub fn one() -> Self {
        Throughput(E::ONE)
    }
    /// The zero throughput (a terminated / blocked path).
    #[inline(always)]
    pub fn zero() -> Self {
        Throughput(E::ZERO)
    }
}

/// Extending a path multiplies throughputs.
impl<E: Field> Mul for Throughput<E> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Throughput(self.0 * rhs.0)
    }
}

impl<E: Field> MulAssign for Throughput<E> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 = self.0 * rhs.0;
    }
}

/// Scale throughput by a dimensionless field factor (e.g. a continuation weight).
impl<E: Field> Mul<E> for Throughput<E> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: E) -> Self {
        Throughput(self.0 * rhs)
    }
}

/// Scale throughput by a dimensionless field factor (e.g. a continuation weight).
impl<E: Field> MulAssign<E> for Throughput<E> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: E) {
        self.0 = self.0 * rhs;
    }
}


/// Divide throughput by a dimensionless field factor (e.g. Russian-roulette
/// continuation probability).
impl<E: Field> Div<E> for Throughput<E> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: E) -> Self {
        Throughput(self.0 / rhs)
    }
}

// ---------------------------------------------------------------------------
// Throughput acting on transported quantities: β · L = L, β · W_e = W_e, etc.
// Both orders are provided so call sites read naturally.
// ---------------------------------------------------------------------------

macro_rules! throughput_scales {
    ($q:ident) => {
        impl<E: Field> Mul<$q<E>> for Throughput<E> {
            type Output = $q<E>;
            #[inline(always)]
            fn mul(self, rhs: $q<E>) -> $q<E> {
                $q(self.0 * rhs.0)
            }
        }
        impl<E: Field> Mul<Throughput<E>> for $q<E> {
            type Output = $q<E>;
            #[inline(always)]
            fn mul(self, rhs: Throughput<E>) -> $q<E> {
                $q(self.0 * rhs.0)
            }
        }
    };
}

throughput_scales!(Radiance);
throughput_scales!(Importance);
throughput_scales!(Emission);

// ---------------------------------------------------------------------------
// Transported quantities accumulate, and may be scaled by a dimensionless field
// weight (a MIS weight, or 1/N). Same shape as math's `Estimate`.
// ---------------------------------------------------------------------------

macro_rules! transported {
    ($q:ident) => {
        impl<E: Field> $q<E> {
            #[inline(always)]
            pub fn zero() -> Self {
                $q(E::ZERO)
            }
        }
        impl<E: Field> Add for $q<E> {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                $q(self.0 + rhs.0)
            }
        }
        impl<E: Field> AddAssign for $q<E> {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                self.0 = self.0 + rhs.0;
            }
        }
        /// Scale by a dimensionless field weight (MIS weight, 1/N, …).
        impl<E: Field> Mul<E> for $q<E> {
            type Output = Self;
            #[inline(always)]
            fn mul(self, rhs: E) -> Self {
                $q(self.0 * rhs)
            }
        }
    };
}

transported!(Radiance);
transported!(Importance);

/// Emitted radiance enters light transport as radiance.
impl<E: Field> From<Emission<E>> for Radiance<E> {
    #[inline(always)]
    fn from(e: Emission<E>) -> Self {
        Radiance(e.0)
    }
}

// ---------------------------------------------------------------------------
// The bridge between the quantity axis and the measure axis.
// ---------------------------------------------------------------------------

impl<E: Field + FromScalar<f32>> BSDF<E> {
    /// The single-bounce Monte Carlo factor `f · cos θ / pdf`, as a dimensionless
    /// [`Throughput`].
    ///
    /// The cosine appears *only* through the change of reference measure: a BSDF
    /// value `f` is an integrand against projected solid angle (`dσ⊥ = cosθ dσ`),
    /// and the supplied directional density `pdf` (against ordinary solid angle)
    /// is re-expressed against σ⊥ by [`PDF::convert`] with [`DirectionalGeom`].
    /// Their ratio is then a measure-free [`Estimate`] (Veach eq. 8.8–8.9), which
    /// we re-tag as throughput:
    ///
    /// ```text
    ///   f / (p_σ / |cosθ|) = f · |cosθ| / p_σ
    /// ```
    #[inline(always)]
    pub fn estimator(self, cos_theta: f32, pdf: PDF<E, SolidAngle>) -> Throughput<E> {
        let pdf_psa: PDF<E, ProjectedSolidAngle> = pdf.convert(DirectionalGeom { cos_theta });
        let integrand: Integrand<E, ProjectedSolidAngle> = Integrand::new(self.0);
        let est: Estimate<E> = integrand / pdf_psa;
        Throughput(*est)
    }
}

// ---------------------------------------------------------------------------
// The measurement: pairing importance with radiance integrates the rendering
// equation's inner product ⟨W_e, L⟩ to a measure-free Estimate (Veach §3.7.1).
// Only this pairing collapses to a film value — `Radiance + Importance` etc. is
// rejected, so you cannot form a measurement from two like quantities.
// ---------------------------------------------------------------------------

impl<E: Field> Mul<Radiance<E>> for Importance<E> {
    type Output = Estimate<E>;
    #[inline(always)]
    fn mul(self, rhs: Radiance<E>) -> Estimate<E> {
        Estimate::new(self.0 * rhs.0)
    }
}

impl<E: Field> Mul<Importance<E>> for Radiance<E> {
    type Output = Estimate<E>;
    #[inline(always)]
    fn mul(self, rhs: Importance<E>) -> Estimate<E> {
        Estimate::new(self.0 * rhs.0)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn role_duals_are_opposite() {
        // Compile-time: Primal and Adjoint are each other's dual.
        fn assert_dual<R: Role, D: Role>()
        where
            R: Role<Dual = D>,
        {
        }
        assert_dual::<Primal, Adjoint>();
        assert_dual::<Adjoint, Primal>();
        // Roles are zero-sized phantoms.
        assert_eq!(std::mem::size_of::<Primal>(), 0);
        assert_eq!(std::mem::size_of::<Adjoint>(), 0);
    }

    #[test]
    fn throughput_extends_and_scales() {
        let beta = Throughput::<f32>::one() * Throughput(0.5) * Throughput(0.4);
        assert_eq!(*beta, 0.2);
        // dimensionless field scaling (e.g. RR)
        assert_eq!(*(Throughput(0.2_f32) / 0.5), 0.4);
    }

    #[test]
    fn throughput_carries_radiance() {
        let l = Radiance(2.0_f32);
        let beta = Throughput(0.25_f32);
        // both orders work and agree
        assert_eq!(*(beta * l), 0.5);
        assert_eq!(*(l * beta), 0.5);
    }

    #[test]
    fn radiance_accumulates_with_mis_weight() {
        let mut sum = Radiance::<f32>::zero();
        sum += Radiance(1.0) * 0.75; // MIS weight
        sum += Radiance(2.0) * 0.25;
        assert_eq!(*sum, 1.25);
    }

    #[test]
    fn emission_enters_as_radiance() {
        let l: Radiance<f32> = Emission(3.0_f32).into();
        assert_eq!(*l, 3.0);
    }

    #[test]
    fn bsdf_estimator_is_f_cos_over_pdf() {
        // f = 0.5 sr^-1, cos = 0.8, p_σ = 0.4 sr^-1  →  0.5 * 0.8 / 0.4 = 1.0
        let f = BSDF(0.5_f32);
        let beta = f.estimator(0.8, PDF::<f32, SolidAngle>::new(0.4));
        assert!((*beta - 1.0).abs() < 1e-6, "got {}", *beta);
    }

    #[test]
    fn measurement_pairs_importance_and_radiance() {
        let we = Importance(4.0_f32);
        let l = Radiance(0.25_f32);
        let est: Estimate<f32> = we * l;
        assert_eq!(*est, 1.0);
    }
}

/// Illegal algebra is rejected at compile time.
///
/// Two radiances cannot be paired into a measurement:
///
/// ```compile_fail
/// use math::prelude::*;
/// let a: Radiance<f32> = Radiance(1.0);
/// let b: Radiance<f32> = Radiance(2.0);
/// let _m: Estimate<f32> = a * b; // ERROR: no `Mul<Radiance> for Radiance`
/// ```
///
/// Radiance and importance cannot be added:
///
/// ```compile_fail
/// use math::prelude::*;
/// let l: Radiance<f32> = Radiance(1.0);
/// let w: Importance<f32> = Importance(2.0);
/// let _ = l + w; // ERROR: no `Add<Importance> for Radiance`
/// ```
///
/// A BSDF value is not itself a throughput — it must go through `estimator`:
///
/// ```compile_fail
/// use math::prelude::*;
/// let f: BSDF<f32> = BSDF(0.5);
/// let l: Radiance<f32> = Radiance(1.0);
/// let _ = f * l; // ERROR: no `Mul<Radiance> for BSDF`
/// ```
#[cfg(doctest)]
struct CompileFailTests;
