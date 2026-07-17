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
//! - `Throughput × Radiance   = Radiance`               (and `Importance`)
//! - `Radiance + Radiance`                              (accumulate; ditto `Importance`)
//! - [`BSDF::estimator`] — the bridge to the measure layer: `f·cos/pdf`
//! - `Importance × Radiance = Integrand` then `/ PDF = Estimate` (the two-stage
//!   measurement, Veach §3.7.1/eq. 4.20 — see the impls near [`MeasurementDensityDim`])

use crate::prelude::*;
use std::{fmt, marker::PhantomData, ops::Deref};

/// Marker for a radiometric quantity (the `Measurable` idea from the original
/// `traits.rs` sketch). Lets generic code bound on "is a radiometric quantity"
/// without enumerating each newtype. Implemented by the legacy newtypes
/// ([`Radiance`], [`Importance`], …) and by the dimension/role/measure-tagged
/// [`Quantity`] carrier (TODO #23 Slice 3).
pub trait Measurable: Copy {
    /// The underlying energy field type (a thermite float vector, e.g.
    /// `Vector<f32>` or `Vector<R>`).
    type Field;
    /// Read the raw value out of the quantity.
    fn value(self) -> Self::Field;
}

/// Transport **role** of a radiometric quantity — which adjoint solution it
/// belongs to (Veach §3.7.3). Role is deliberately *not* a dimension: importance
/// `W_e` and radiance `L` carry the *same* dimensions, so the radiance/importance
/// distinction can only be a separate zero-cost phantom. It lets the measurement
/// bridge `⟨W_e, L⟩ = Adjoint × Prime → Estimate` (Veach §3.7.1) be typed while a
/// nonsensical `L · L` is rejected. Consumed by the Slice 3 carrier (TODO #23).
pub trait Role: Copy + Default {
    /// The opposite transport role (`Prime ↔ Adjoint`) — the partner a
    /// measurement pairs this role with.
    type Dual: Role;
}

/// The **primal** transport solution: radiance / the measurement carried toward
/// the sensor.
#[derive(Copy, Clone, Default, PartialEq, Eq, Debug)]
pub struct Prime;
/// The **adjoint** transport solution: importance carried from the sensor
/// (particle / light tracing).
#[derive(Copy, Clone, Default, PartialEq, Eq, Debug)]
pub struct Adjoint;

impl Role for Prime {
    type Dual = Adjoint;
}
impl Role for Adjoint {
    type Dual = Prime;
}

/// Generates a `pub struct Name<E>(pub E)` newtype with `Deref<Target = E>`,
/// `Measurable`, and the usual derives, so the per-quantity blocks below only have
/// to spell out the algebra that is unique to them.
macro_rules! quantity {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(Copy, Clone, Debug)]
        pub struct $name<E>(pub E);

        impl<E> Deref for $name<E> {
            type Target = E;
            #[inline(always)]
            fn deref(&self) -> &E {
                &self.0
            }
        }

        impl<E: Copy> Measurable for $name<E> {
            type Field = E;
            #[inline(always)]
            fn value(self) -> E {
                self.0
            }
        }
    };
}

// NOTE(task #23): `Radiance`, `Importance`, and `Irradiance` are no longer
// bespoke newtypes — they are migrated onto the dimension/role/measure-tagged
// [`Quantity`] carrier and defined as type aliases further down (see the
// "carrier" section), alongside `BSDF`. `Emission` is gone entirely: emitted
// radiance *is* radiance, so `Material::emission` now returns [`Radiance`]
// directly. Only [`Throughput`] remains a hand-written newtype — it is genuinely
// dimensionless, roleless, and measureless (a path weight, not a point in the
// (dimension, role, measure) space), so it has no honest carrier tags.
quantity!(
    /// Dimensionless path throughput `β`: the running product of `f·cos/pdf`
    /// ratios along a path. Carries no units — it scales a transported quantity.
    Throughput
);

// ---------------------------------------------------------------------------
// Throughput: multiplicative, dimensionless.
// ---------------------------------------------------------------------------

impl<E: FloatVector> Throughput<E> {
    /// The multiplicative identity throughput (a fresh path carries `β = 1`).
    #[inline(always)]
    pub fn one() -> Self {
        Throughput(<E as NumericVector>::ONE)
    }
    /// The zero throughput (a terminated / blocked path).
    #[inline(always)]
    pub fn zero() -> Self {
        Throughput(<E as NumericVector>::ZERO)
    }
}

/// Extending a path multiplies throughputs.
impl<E: FloatVector> Mul for Throughput<E> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Throughput(self.0 * rhs.0)
    }
}

impl<E: FloatVector> MulAssign for Throughput<E> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 = self.0 * rhs.0;
    }
}

/// Scale throughput by a dimensionless field factor (e.g. a continuation weight).
impl<E: FloatVector> Mul<E> for Throughput<E> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: E) -> Self {
        Throughput(self.0 * rhs)
    }
}

/// Scale throughput by a dimensionless field factor (e.g. a continuation weight).
impl<E: FloatVector> MulAssign<E> for Throughput<E> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: E) {
        self.0 = self.0 * rhs;
    }
}

/// Divide throughput by a dimensionless field factor (e.g. Russian-roulette
/// continuation probability).
impl<E: FloatVector> Div<E> for Throughput<E> {
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

// `Throughput` scaling a carrier-borne transported quantity (`β · L = L`,
// `β · W_e = W_e`) is provided generically further down, on the [`Quantity`]
// carrier itself (`Mul<Throughput> for Quantity`), so it covers every
// transported alias at once without a per-quantity macro. The carrier also
// supplies accumulation (`Add`/`AddAssign` with `SameDimension`) and
// dimensionless field scaling (`Mul<T>`), which the old `transported!` macro
// hand-rolled per newtype.

// ===========================================================================
// The dimension/role/measure-tagged carrier (TODO #23 Slice 3).
//
// `Quantity<T, D, M>` is the *generated* form the bespoke newtypes above are
// migrating toward: it carries, on zero-cost phantom tags, the base-dimension
// exponent algebra `D` (`dimension::*`), the transport [`Role`] `R`, and the
// reference [`Measure`] `M`. A mismatched dimension or measure becomes a compile
// error, and the bespoke `Mul`/`Div`/measurement impls *derive* from the tags.
//
// Migration is incremental: only `BSDF` is on the carrier so far (the others stay
// newtypes so PT/LT keep compiling). The tags are markers, so a `Quantity` is the
// same size/codegen as the bare `T` it wraps (asserted in the tests).
// ===========================================================================

/// A radiometric value of field `T`, tagged with its dimension `D`
/// ([`dimension::Dimension`]), transport [`Role`] `R`, and reference [`Measure`]
/// `M`. All three tags are zero-sized phantoms — the value is just a `T`.
///
/// `PhantomData<fn() -> (D,  M)>` (not `*const _`) keeps the carrier
/// `Send + Sync` for the threaded renderer while staying invariant in the tags.
pub struct Quantity<T: FloatVector, D: Dimension, M: Measure> {
    v: T,
    tags: PhantomData<fn() -> (D, M)>,
}

impl<T: FloatVector, D: Dimension, M: Measure> Quantity<T, D, M> {
    /// Wrap a raw field value with the (inferred) dimension/role/measure tags.
    #[inline(always)]
    pub fn new(v: T) -> Self {
        Self {
            v,
            tags: PhantomData,
        }
    }

    /// The additive identity (a zero radiance / importance accumulator).
    #[inline(always)]
    pub fn zero() -> Self {
        Self::new(<T as NumericVector>::ZERO)
    }

    /// Re-tag this value with the canonical (normalized) form of its dimension —
    /// a zero-cost coercion (the value is untouched). This is the "lazy"
    /// normalization seam: build dimensions with the readable `Product`/base
    /// types, then `normalize()` before storing into a canonically-typed field.
    #[inline(always)]
    pub fn normalize(self) -> Quantity<T, Normalized<D>, M>
    where
        D: Normalize,
        Normalized<D>: Dimension,
    {
        Quantity::new(self.v)
    }

    /// View this value as an [`Integrand`] against its measure `M` — the bridge
    /// into the Monte Carlo estimator (`Integrand / PDF → Estimate`). Preserves
    /// this quantity's dimension `D` onto the integrand (TODO #27): `f`'s
    /// dimension doesn't vanish just because it's about to be integrated — see
    /// [`Integrand`]'s `Div<PDF>` for where `D` combines with the measure.
    #[inline(always)]
    pub fn as_integrand(self) -> Integrand<T, M, D> {
        Integrand::new(self.v)
    }
}

// Manual Clone/Copy/Debug so the tags need not be `Copy`/`Debug` themselves
// (e.g. the dimension markers don't implement `PartialEq`): the phantom is always
// `Copy`, and only the value `T` participates.
impl<T: FloatVector, D: Dimension, M: Measure> Clone for Quantity<T, D, M> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: FloatVector, D: Dimension, M: Measure> Copy for Quantity<T, D, M> {}

impl<T: FloatVector + fmt::Debug, D: Dimension, M: Measure> fmt::Debug for Quantity<T, D, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Quantity({:?})", self.v)
    }
}

impl<T: FloatVector, D: Dimension, M: Measure> Deref for Quantity<T, D, M> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &T {
        &self.v
    }
}

impl<T: FloatVector, D: Dimension, M: Measure> From<T> for Quantity<T, D, M> {
    #[inline(always)]
    fn from(v: T) -> Self {
        Self::new(v)
    }
}

impl<T: FloatVector, D: Dimension, M: Measure> Measurable for Quantity<T, D, M> {
    type Field = T;
    #[inline(always)]
    fn value(self) -> T {
        self.v
    }
}

/// Scale a tagged quantity by a dimensionless field weight (a MIS weight, `1/N`,
/// a cosine already folded in elsewhere) — tags unchanged.
impl<T: FloatVector, D: Dimension, M: Measure> Mul<T> for Quantity<T, D, M> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: T) -> Self {
        Quantity::new(self.v * rhs)
    }
}
impl<T: FloatVector, D: Dimension, M: Measure> Div<T> for Quantity<T, D, M> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: T) -> Self {
        Quantity::new(self.v / rhs)
    }
}

/// Accumulate two quantities of the *same* dimension (up to normalization), role,
/// and measure. The output is tagged with the canonical (normalized) dimension —
/// the carrier's combining op normalizes its own output (the practical "lazy"
/// rule), so accumulation is a renormalization seam.
impl<T, D1, D2, M> Add<Quantity<T, D2, M>> for Quantity<T, D1, M>
where
    T: FloatVector,
    D1: Dimension + Normalize,
    D2: Dimension + SameDimension<D1>,
    Normalized<D1>: Dimension,
    M: Measure,
{
    type Output = Quantity<T, Normalized<D1>, M>;
    #[inline(always)]
    fn add(self, rhs: Quantity<T, D2, M>) -> Self::Output {
        Quantity::new(self.v + rhs.v)
    }
}

/// In-place accumulation `l += l2` of two quantities of the *same* dimension (up
/// to normalization), role, and measure. Unlike `Add` (which retags its output to
/// the canonical dimension), `AddAssign` keeps `self`'s dimension tag `D1` — it
/// mutates the value in place, so the type cannot change.
impl<T, D1, D2, M> AddAssign<Quantity<T, D2, M>> for Quantity<T, D1, M>
where
    T: FloatVector,
    D1: Dimension,
    D2: Dimension + SameDimension<D1>,
    M: Measure,
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: Quantity<T, D2, M>) {
        self.v = self.v + rhs.v;
    }
}

// ---------------------------------------------------------------------------
// `Throughput` carries a transported quantity: `β · L = L`, `β · W_e = W_e`.
// Provided once on the carrier (covering every transported alias), in both orders
// so call sites read naturally. The tags are preserved — throughput is a
// dimensionless/roleless/measureless scalar weight.
// ---------------------------------------------------------------------------

impl<T: FloatVector, D: Dimension, M: Measure> Mul<Throughput<T>> for Quantity<T, D, M> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Throughput<T>) -> Self {
        Quantity::new(self.v * rhs.0)
    }
}
impl<T: FloatVector, D: Dimension, M: Measure> Mul<Quantity<T, D, M>> for Throughput<T> {
    type Output = Quantity<T, D, M>;
    #[inline(always)]
    fn mul(self, rhs: Quantity<T, D, M>) -> Quantity<T, D, M> {
        Quantity::new(self.0 * rhs.v)
    }
}

// ---------------------------------------------------------------------------
// `Density` — a radiometric quantity whose dimension is *derived* from its
// measure, not asserted alongside it (TODO #26). Veach App. 3.B: a quantity is a
// Radon–Nikodym derivative `f = dQ/dϱ`, so `dim(f) = dim(Q) − dim(ϱ)`. Here `Num`
// is the numerator dimension `dim(Q)` (the energy/sensor content) and `M`'s
// [`Measure::Dim`] is `dim(ϱ)`; the carrier dimension is their normalized
// quotient, so it can't drift from the measure `M` the quantity references.
// ---------------------------------------------------------------------------

/// A radiometric **density** of `Num` per measure `M`: the carrier dimension is
/// derived as `Normalized(Num ÷ M::Dim)`. `Num` is the numerator dimension
/// ([`PowerDim`] for radiance/irradiance, [`Dimensionless`] for a BSDF), `R` the
/// transport [`Role`], `M` the reference [`Measure`].
pub type Density<T, Num, M> =
    Quantity<T, Normalized<Product<Num, Reciprocal<<M as Measure>::Dim>>>, M>;

// ---------------------------------------------------------------------------
// The transported-quantity aliases (TODO #23/#26). Radiance and importance share
// the same numerator (`Φ`) and measure (throughput) and differ only in transport
// [`Role`]; irradiance drops the solid-angle axis (measure = area). Defined via
// [`Density`], so each dimension is *derived* `Φ ÷ dim(measure)` — Radiance
// `Φ ÷ (A·σ⊥) = Φ·A⁻¹·Ω⁻¹`, Irradiance `Φ ÷ A = Φ·A⁻¹` — and normalized, so the
// carrier's `Add` (which retags to canonical) lands back on the same alias type.
// ---------------------------------------------------------------------------

/// Radiance `L` (W·m⁻²·sr⁻¹) — the primal quantity transported toward the camera
/// (Veach §3.4.3). Emitted radiance `L_e` is the same type (no separate
/// `Emission`). What a path tracer accumulates.
pub type Radiance<E> = Density<E, PowerDim, ThroughputMeasure>;

/// Importance `W_e` — sensor flux responsivity, the adjoint of radiance (Veach
/// §3.7.3, eq. 4.19: `W_e = dS/dΦ`, a unitless-sensor response `S` per unit
/// radiant flux `Φ`). Dimension `Φ⁻¹` (TODO #27) — *not* [`Radiance`]'s
/// dimension, despite both being adjoint/primal partners in the same
/// measurement: importance is a density against the *energy* measure
/// [`PowerMeasure`], radiance against [`ThroughputMeasure`]. Pairing the two
/// (`Importance × Radiance`) is therefore a genuine Radon–Nikodym chain-rule
/// cancellation, not a same-dimension multiply — see the measurement impls
/// below.
pub type Importance<E> = Density<E, Dimensionless, PowerMeasure>;

/// Irradiance `E` (W·m⁻²) — radiance integrated over the projected hemisphere
/// (Veach §3.4.2). An integrand against area.
pub type Irradiance<E> = Density<E, PowerDim, Area>;

// ---------------------------------------------------------------------------
// `BSDF` — the first quantity on the carrier. A BSDF value `f_s` has dimension
// `Ω⁻¹` (`BsdfDim`) and is an integrand against **projected** solid angle; it is
// a primal-side quantity.
// ---------------------------------------------------------------------------

/// A bidirectional scattering distribution value `f_s` (units sr⁻¹), Veach §3.6
/// — now a [`Quantity`] carrier instantiation (`Ω⁻¹`, primal, per projected solid
/// angle). Combine with a cosine and a directional density via [`BSDF::estimator`]
/// to get a dimensionless [`Throughput`] factor.
pub type BSDF<E> = Density<E, Dimensionless, ProjectedSolidAngle>;

impl<E, T> BSDF<E>
where
    E: FloatVector<Element = T>,
    T: FloatElement + From<f32>,
{
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
        let est: Estimate<E> = self.as_integrand() / pdf_psa;
        Throughput(*est)
    }
}

// ---------------------------------------------------------------------------
// The measurement: pairing importance with radiance is a two-stage
// Radon–Nikodym chain rule (Veach eq. 4.20, §3.7.1), not a same-dimension
// multiply-and-tag:
//
//   stage 1:  (dS/dΦ)·(dΦ/dμ) = dS/dμ         — the energy measure Φ cancels
//   stage 2:  ∫ (dS/dμ) dμ = S                — the μ-integration (Integrand/PDF)
//
// `Importance × Radiance` performs stage 1 only, landing on an `Integrand`
// against `ThroughputMeasure` (the sensor-response density `dS/dμ`, dimension
// `Reciprocal<ThroughputMeasure::Dim>` — Φ has cancelled). Stage 2 is the
// ordinary `Integrand / PDF<_, ThroughputMeasure> -> Estimate` division
// (`pdf.rs`): dividing by the ray-space sampling density performs the
// μ-integration, and the dimensions cancel to `Nil` by construction (their
// product is exactly `ThroughputMeasure::Dim · Reciprocal<ThroughputMeasure::Dim>`).
// Only this pairing collapses toward a film value — `Radiance + Importance`
// etc. is still rejected, so a measurement can't be formed from two like
// quantities.
// ---------------------------------------------------------------------------

/// Dimension of the stage-1 sensor-response density `dS/dμ` (Veach eq. 4.20):
/// `Importance::Dim · Radiance::Dim`, which is `Reciprocal<ThroughputMeasure::Dim>`
/// since the energy measure `Φ` cancels between them.
pub type MeasurementDensityDim = Normalized<Reciprocal<<ThroughputMeasure as Measure>::Dim>>;

impl<E: FloatVector> Mul<Radiance<E>> for Importance<E> {
    type Output = Integrand<E, ThroughputMeasure, MeasurementDensityDim>;
    #[inline(always)]
    fn mul(self, rhs: Radiance<E>) -> Self::Output {
        Integrand::new(*self * *rhs)
    }
}

impl<E: FloatVector> Mul<Importance<E>> for Radiance<E> {
    type Output = Integrand<E, ThroughputMeasure, MeasurementDensityDim>;
    #[inline(always)]
    fn mul(self, rhs: Importance<E>) -> Self::Output {
        Integrand::new(*self * *rhs)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    type Vf = Vector<f32>;
    /// splat a scalar into the 1-lane field type.
    fn s(x: f32) -> Vf {
        Vf::splat(x)
    }
    /// read lane 0 back out (the quantity newtypes no longer derive `PartialEq`).
    fn e(v: Vf) -> f32 {
        v.extract::<0>()
    }

    // --- the Quantity carrier (Slice 3) ----------------------------------

    #[test]
    fn carrier_is_zero_cost_and_threadsafe() {
        // Lane-generic over the field: scalar f32 and a SIMD register both work.
        type Lanes = <thermite::backend::scalar::Scalar as thermite::simd::Simd>::f32x4;
        // The tags are phantoms: a tagged quantity is the same size as its field.
        assert_eq!(
            std::mem::size_of::<BSDF<Vf>>(),
            std::mem::size_of::<Vf>()
        );
        assert_eq!(
            std::mem::size_of::<BSDF<Vector<Lanes>>>(),
            std::mem::size_of::<Vector<Lanes>>()
        );
        // Send + Sync for the threaded renderer.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BSDF<Vf>>();
        assert_send_sync::<BSDF<Vector<Lanes>>>();
    }

    #[test]
    fn bsdf_estimator_matches_f_cos_over_pdf() {
        // f·cos/p_σ via the measure-correct bridge.
        let f = BSDF::<Vf>::new(s(0.5));
        let pdf: PDF<Vf, SolidAngle> = PDF::new(s(0.25));
        let beta = f.estimator(0.4, pdf);
        // f·cos/p = 0.5·0.4/0.25 = 0.8
        assert!((e(*beta) - 0.8).abs() < 1e-6, "got {}", e(*beta));
    }

    #[test]
    fn carrier_scales_and_derefs() {
        let f = BSDF::<Vf>::new(s(0.6));
        assert_eq!(e(*f), 0.6); // Deref
        assert_eq!(e(*(f * s(0.5))), 0.3); // Mul<V>
        assert_eq!(e(*(f / s(2.0))), 0.3); // Div<V>
        assert_eq!(e(f.value()), 0.6); // Measurable
    }

    #[test]
    fn carrier_normalize_is_value_noop() {
        // `normalize()` is a zero-cost retag — value untouched, dimension becomes
        // the canonical form.
        let f = BSDF::<Vf>::new(s(0.7));
        let n: Quantity<Vf, Normalized<BsdfDim>, ProjectedSolidAngle> = f.normalize();
        assert_eq!(e(*n), 0.7);
    }

    #[test]
    fn role_duals_are_opposite() {
        // Compile-time: Prime and Adjoint are each other's dual.
        fn assert_dual<D: Role<Dual = E>, E: Role>() {}
        assert_dual::<Prime, Adjoint>();
        assert_dual::<Adjoint, Prime>();
        // Roles are zero-sized phantoms.
        assert_eq!(std::mem::size_of::<Prime>(), 0);
        assert_eq!(std::mem::size_of::<Adjoint>(), 0);
    }

    #[test]
    fn throughput_extends_and_scales() {
        let beta = Throughput::<Vf>::one() * Throughput(s(0.5)) * Throughput(s(0.4));
        assert_eq!(e(*beta), 0.2);
        // dimensionless field scaling (e.g. RR)
        assert_eq!(e(*(Throughput(s(0.2)) / s(0.5))), 0.4);
    }

    #[test]
    fn throughput_carries_radiance() {
        let l = Radiance::new(s(2.0));
        let beta = Throughput(s(0.25));
        // both orders work and agree
        assert_eq!(e(*(beta * l)), 0.5);
        assert_eq!(e(*(l * beta)), 0.5);
    }

    #[test]
    fn radiance_accumulates_with_mis_weight() {
        let mut sum = Radiance::<Vf>::zero();
        sum += Radiance::new(s(1.0)) * s(0.75); // MIS weight
        sum += Radiance::new(s(2.0)) * s(0.25);
        assert_eq!(e(*sum), 1.25);
    }

    #[test]
    fn bsdf_estimator_is_f_cos_over_pdf() {
        // f = 0.5 sr^-1, cos = 0.8, p_σ = 0.4 sr^-1  →  0.5 * 0.8 / 0.4 = 1.0
        let f = BSDF::new(s(0.5));
        let beta = f.estimator(0.8, PDF::<Vf, SolidAngle>::new(s(0.4)));
        assert!((e(*beta) - 1.0).abs() < 1e-6, "got {}", e(*beta));
    }

    #[test]
    fn measurement_pairs_importance_and_radiance() {
        // Two-stage Radon–Nikodym chain rule (Veach eq. 4.20): stage 1 cancels
        // the energy measure Φ (Importance × Radiance → Integrand), stage 2
        // integrates against ray space (÷ PDF<_, ThroughputMeasure> → Estimate).
        let we = Importance::new(s(4.0));
        let l = Radiance::new(s(0.25));
        let ds_dmu = we * l;
        let p_ray = PDF::<Vf, ThroughputMeasure>::new(s(1.0));
        let est: Estimate<Vf> = ds_dmu / p_ray;
        assert_eq!(e(*est), 1.0);
    }

    #[test]
    fn measurement_dim_identity() {
        // TODO #27's named verification: Importance::Dim · Radiance::Dim is the
        // reciprocal of the throughput measure's own dimension (the Φ measure
        // cancels between the two, Veach eq. 4.19/4.20).
        assert_same::<
            Product<Normalized<Product<Dimensionless, Reciprocal<PowerDim>>>, RadianceDim>,
            Reciprocal<<ThroughputMeasure as Measure>::Dim>,
        >();
    }

    // --- #26: dimension derived from measure (Density) -------------------

    /// Compile-time assertion that two dimensions are dimensionally equal.
    fn assert_same<A: SameDimension<B>, B>() {}

    #[test]
    fn density_derives_the_handwritten_dimensions() {
        // The whole point of #26: `Num ÷ M::Dim` reduces to the hand-written
        // `*Dim`, grounded in Veach App. 3.B (`dim(f) = dim(Q) − dim(ϱ)`).
        // Radiance: Φ ÷ dim(throughput) = Φ ÷ (A·σ⊥) = Φ·A⁻¹·Ω⁻¹.
        assert_same::<
            Product<PowerDim, Reciprocal<<ThroughputMeasure as Measure>::Dim>>,
            RadianceDim,
        >();
        // Irradiance: Φ ÷ dim(area) = Φ ÷ A = Φ·A⁻¹.
        assert_same::<Product<PowerDim, Reciprocal<<Area as Measure>::Dim>>, IrradianceDim>();
        // BSDF: 1 ÷ dim(projected solid angle) = Ω⁻¹.
        assert_same::<
            Product<Dimensionless, Reciprocal<<ProjectedSolidAngle as Measure>::Dim>>,
            BsdfDim,
        >();
    }

    #[test]
    fn solid_angle_variants_share_a_dimension() {
        // The `Measure → Dim` map is many-to-one: `SolidAngle` and
        // `ProjectedSolidAngle` are distinct measures (the cosine RN factor
        // relates them) yet both carry dimension `Ω`. (That the *measures* stay
        // distinct is asserted by the compile_fail in `CompileFailTests`.)
        assert_same::<<SolidAngle as Measure>::Dim, <ProjectedSolidAngle as Measure>::Dim>();
    }
}

/// Illegal algebra is rejected at compile time.
///
/// Two radiances cannot be paired into a measurement:
///
/// ```compile_fail
/// use math::prelude::*;
/// let a: Radiance<f32> = Radiance::new(1.0);
/// let b: Radiance<f32> = Radiance::new(2.0);
/// let _m: Estimate<f32> = a * b; // ERROR: no `Mul<Radiance> for Radiance`
/// ```
///
/// Radiance and importance cannot be added:
///
/// ```compile_fail
/// use math::prelude::*;
/// let l: Radiance<f32> = Radiance::new(1.0);
/// let w: Importance<f32> = Importance::new(2.0);
/// let _ = l + w; // ERROR: no `Add<Importance> for Radiance`
/// ```
///
/// A BSDF value is not itself a throughput — it must go through `estimator`:
///
/// ```compile_fail
/// use math::prelude::*;
/// let f: BSDF<f32> = BSDF::new(0.5);
/// let l: Radiance<f32> = Radiance::new(1.0);
/// let _ = f * l; // ERROR: no `Mul<Radiance> for BSDF`
/// ```
///
/// `SolidAngle` and `ProjectedSolidAngle` share dimension `Ω` (TODO #26) but are
/// **distinct measures** — the `Measure → Dim` map is many-to-one and not
/// invertible, so a density against one cannot stand in for the other:
///
/// ```compile_fail
/// use math::prelude::*;
/// let p: PDF<f32, SolidAngle> = PDF::new(1.0);
/// let _q: PDF<f32, ProjectedSolidAngle> = p; // ERROR: different measures
/// ```
///
/// Pairing importance with radiance is a two-*stage* Radon–Nikodym chain rule
/// (TODO #27): stage 1 (`Importance × Radiance`) only cancels the energy
/// measure `Φ`, landing on an `Integrand` — not yet a film-ready `Estimate`.
/// Skipping stage 2 (the `÷ PDF<_, ThroughputMeasure>` μ-integration) is
/// rejected:
///
/// ```compile_fail
/// use math::prelude::*;
/// let we: Importance<f32> = Importance::new(1.0);
/// let l: Radiance<f32> = Radiance::new(1.0);
/// let _m: Estimate<f32> = we * l; // ERROR: `we * l` is an `Integrand`, not an `Estimate`
/// ```
#[cfg(doctest)]
struct CompileFailTests;
