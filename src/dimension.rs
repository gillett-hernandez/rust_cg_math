//! Compile-time **dimensional algebra** — the base-dimension structs that
//! radiometric quantities are built from (TODO #23).
//!
//! This mirrors the *measure* layer in [`crate::traits`]: just as `Area =
//! ProductMeasure<Length, Length>` composes measures, a dimension composes base
//! axes. The difference is that each base axis carries a **signed exponent** (a
//! [`typenum::Integer`]) *on the type itself*, so:
//!
//! - multiplying the **same** axis adds exponents — `Length<A> · Length<B> =
//!   Length<Sum<A,B>>` — and dividing subtracts them; reciprocal negates;
//! - composing **different** axes uses [`Product`] (a named type, like
//!   `ProductMeasure` — e.g. `RadianceDim = Product<Power<P1>, …>`).
//!
//! Base axes (the set is open: a new axis is just another `base_dimension!`
//! struct, nothing central to edit):
//!
//! | axis | base struct | meaning |
//! |---|---|---|
//! | length `L`       | [`Length`]     | geometry; `Area = Length<P2>` |
//! | power/energy `Φ` | [`Power`]      | the radiant unit transported from a light |
//! | solid angle `Ω`  | [`SolidAngle`] | directions (projected variant is a measure, not a base dim) |
//! | wavelength `Λ`   | [`Wavelength`] | spectral selection (the axis #20 surfaced) |
//!
//! These names deliberately shadow the *measure* names (`Length`, `SolidAngle` in
//! [`crate::traits`]); they live in this module and the prelude re-exports only
//! the composed `*Dim` aliases and the combinators, not the bare base structs, so
//! there is no glob collision. Refer to the bases as `dimension::Length<N>` etc.
//!
//! ## What is and isn't normalized
//!
//! Same-axis exponents merge (`Length<P1> · Length<P1> = Length<P2>`, nominally
//! `AreaDim`). But **cross-axis** composition does not normalize: the `*`
//! operator only merges a base with the *same* base (dispatching merge-vs-nest
//! generically is impossible under coherence), so different axes are combined by
//! naming a [`Product`] type. Two cross-axis [`Product`]s that are dimensionally
//! equal but built in a different order/shape are therefore distinct types; a
//! Product-normalization pass is future work if a quantity ever needs to compare
//! them.

use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};

use typenum::{Diff, Integer, Negate, Sum};
// comparison machinery for the sorted-list normalization (Slice 1.5).
use typenum::{Cmp, Compare, Equal, Greater, Less, Unsigned, Z0};
use typenum::{U0, U1, U2, U3};
// the small signed-integer literals used by the dimension aliases below.
use typenum::{N1, N2, P1, P2};

/// Marker for a compile-time dimension. The exponent(s) live on the type (the
/// typenum parameter of each base axis), not in a central tuple — so the set of
/// axes stays open. Implemented by the base axes, by [`Product`] (when both
/// children are dimensions, so products nest), and by [`Dimensionless`].
pub trait Dimension: Copy + Default {}

/// The reciprocal of a dimension (negate every exponent). [`Div`] across
/// different axes would compose `Self · Recip<RHS>`; here it powers the
/// dimensionless-numerator case `1 / D`.
pub trait Recip {
    /// the reciprocal dimension
    type Output: Dimension;
}

/// The product of two **different-axis** dimensions, mirroring
/// [`crate::traits::ProductMeasure`]. Either child may itself be a `Product`, so
/// dimensions nest arbitrarily. (Same-axis products are merged by the base-axis
/// [`Mul`] instead — see the module docs.)
pub struct Product<D0, D1>(PhantomData<(D0, D1)>);

impl<D0, D1> Default for Product<D0, D1> {
    #[inline(always)]
    fn default() -> Self {
        Self(PhantomData)
    }
}
impl<D0, D1> Clone for Product<D0, D1> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<D0, D1> Copy for Product<D0, D1> {}

impl<D0: Dimension, D1: Dimension> Dimension for Product<D0, D1> {}

impl<D0: Recip, D1: Recip> Recip for Product<D0, D1> {
    type Output = Product<D0::Output, D1::Output>;
}

impl<D0: fmt::Debug + Default, D1: fmt::Debug + Default> fmt::Debug for Product<D0, D1> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Product<{:?}, {:?}>", D0::default(), D1::default())
    }
}

/// Generates a base-dimension struct `Name<N: Integer>` carrying its exponent on
/// the type, plus boilerplate (`Default/Clone/Copy/Debug`), the [`Dimension`]
/// marker, the same-axis algebra (`Mul` adds exponents, `Div` subtracts, `Recip`
/// negates), and the normalization hooks ([`HasAxis`] + [`Normalize`]) keyed on a
/// freshly-generated [`Axis`] marker `$axis` with [`Unsigned`] key `$key`.
macro_rules! base_dimension {
    ($(#[$m:meta])* $name:ident, $axis:ident, $key:ty, $sym:literal) => {
        #[doc = concat!("Axis marker for the [`", stringify!($name), "`] base dimension.")]
        #[derive(Default, Clone, Copy, Debug)]
        pub struct $axis;
        impl Axis for $axis {
            type Key = $key;
            const SYMBOL: &'static str = $sym;
        }

        $(#[$m])*
        pub struct $name<N: Integer>(PhantomData<N>);

        impl<N: Integer> Default for $name<N> {
            #[inline(always)]
            fn default() -> Self {
                Self(PhantomData)
            }
        }
        impl<N: Integer> Clone for $name<N> {
            #[inline(always)]
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<N: Integer> Copy for $name<N> {}
        impl<N: Integer> fmt::Debug for $name<N> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}<{}>", stringify!($name), <N as Integer>::I32)
            }
        }

        impl<N: Integer> Dimension for $name<N> {}

        /// Same-axis product: exponents add.
        impl<N, M> Mul<$name<M>> for $name<N>
        where
            N: Integer + Add<M>,
            M: Integer,
            Sum<N, M>: Integer,
        {
            type Output = $name<Sum<N, M>>;
            #[inline(always)]
            fn mul(self, _: $name<M>) -> Self::Output {
                $name::default()
            }
        }

        /// Same-axis quotient: exponents subtract.
        impl<N, M> Div<$name<M>> for $name<N>
        where
            N: Integer + Sub<M>,
            M: Integer,
            Diff<N, M>: Integer,
        {
            type Output = $name<Diff<N, M>>;
            #[inline(always)]
            fn div(self, _: $name<M>) -> Self::Output {
                $name::default()
            }
        }

        impl<N> Recip for $name<N>
        where
            N: Integer + Neg,
            Negate<N>: Integer,
        {
            type Output = $name<Negate<N>>;
        }

        impl<N: Integer> HasAxis for $name<N> {
            type Ax = $axis;
            type Exp = N;
        }

        /// Normalize a base axis: insert its single `(axis, exponent)` term into
        /// the empty list (which drops it when the exponent is zero).
        impl<N: Integer> Normalize for $name<N>
        where
            Nil: Insert<$axis, N>,
        {
            type Output = <Nil as Insert<$axis, N>>::Output;
        }
    };
}

base_dimension!(
    /// Length axis `L`.
    Length, LengthAxis, U0, "L"
);
base_dimension!(
    /// Radiant power / energy axis `Φ`.
    Power, PowerAxis, U1, "Φ"
);
base_dimension!(
    /// Solid-angle axis `Ω`.
    SolidAngle, SolidAngleAxis, U2, "Ω"
);
base_dimension!(
    /// Wavelength axis `Λ`.
    Wavelength, WavelengthAxis, U3, "Λ"
);

/// The dimensionless unit `1` — path throughput, MIS weights, the result of a
/// fully-cancelled ratio. Multiplying by it is the identity; `1 / D` is the
/// reciprocal of `D`.
#[derive(Default, Clone, Copy)]
pub struct Dimensionless;

impl fmt::Debug for Dimensionless {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dimensionless")
    }
}
impl Dimension for Dimensionless {}
impl Recip for Dimensionless {
    type Output = Dimensionless;
}
/// `1 · D = D`.
impl<RHS: Dimension> Mul<RHS> for Dimensionless {
    type Output = RHS;
    #[inline(always)]
    fn mul(self, rhs: RHS) -> RHS {
        rhs
    }
}
/// `1 / D = D⁻¹`.
impl<RHS: Recip> Div<RHS> for Dimensionless {
    type Output = <RHS as Recip>::Output;
    #[inline(always)]
    fn div(self, _: RHS) -> Self::Output {
        Default::default()
    }
}

// ===========================================================================
// Normalization — the equality backbone (TODO #23, Slice 1.5).
//
// `Mul`/`Div` above keep *readable* `Product`/base types and deliberately do not
// canonicalize cross-axis composition (coherence forbids generically dispatching
// merge-vs-nest). Normalization reduces any `Dimension` to a canonical, sorted
// association list `axis ⇒ exponent`, so two dimensionally-equal types become the
// *same* type — making dimensional equality ([`SameDimension`]) and
// dimensionlessness ([`IsDimensionless`]) decidable at compile time, while the
// set of axes stays open. This is a pure comparison/coercion layer (the "lazy"
// design): it is invoked only at boundaries (a carrier's `Add`, a dimensionless
// assert, a `normalize()` retag), never eagerly inside `Mul`/`Div`.
// ===========================================================================

/// A distinct dimension **axis**, totally ordered by its unique [`Unsigned`]
/// `Key`. The registry is open — add an axis by adding a `base_dimension!` with
/// the next free key. Current registry: `LengthAxis=U0`, `PowerAxis=U1`,
/// `SolidAngleAxis=U2`, `WavelengthAxis=U3`.
pub trait Axis: Default + Copy {
    /// unique, totally-ordered key over the axis registry
    type Key: Unsigned;
    /// short symbol for `Debug` (`"L"`, `"Φ"`, …)
    const SYMBOL: &'static str;
}

/// A base dimension that lives on a single [`Axis`] with a signed exponent.
/// (Generated for each `base_dimension!` struct.)
pub trait HasAxis {
    /// the axis this base occupies
    type Ax: Axis;
    /// its signed exponent
    type Exp: Integer;
}

/// Canonical empty dimension — the normalized form of [`Dimensionless`] and of
/// any fully-cancelled expression.
#[derive(Default, Clone, Copy)]
pub struct Nil;

/// One `axis ⇒ exponent` entry of a normalized dimension. Invariant, maintained
/// by construction: the exponent `N` is never zero.
pub struct Term<A: Axis, N: Integer>(PhantomData<(A, N)>);

/// A non-empty normalized dimension: a [`Term`] cons-ed onto a strictly
/// key-ascending `Tail`.
pub struct Cons<H, T>(PhantomData<(H, T)>);

impl<A: Axis, N: Integer> Default for Term<A, N> {
    #[inline(always)]
    fn default() -> Self {
        Self(PhantomData)
    }
}
impl<A: Axis, N: Integer> Clone for Term<A, N> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<A: Axis, N: Integer> Copy for Term<A, N> {}

impl<H, T> Default for Cons<H, T> {
    #[inline(always)]
    fn default() -> Self {
        Self(PhantomData)
    }
}
impl<H, T> Clone for Cons<H, T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<H, T> Copy for Cons<H, T> {}

// The canonical forms are first-class dimensions too, so a carrier can be tagged
// with `Normalized<D>` directly (Slice 3).
impl Dimension for Nil {}
impl<A: Axis, N: Integer> Dimension for Term<A, N> {}
impl<H, T> Dimension for Cons<H, T> {}

impl fmt::Debug for Nil {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "1")
    }
}
impl<A: Axis, N: Integer> fmt::Debug for Term<A, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}^{}", A::SYMBOL, <N as Integer>::I32)
    }
}
impl<H: fmt::Debug + Default, T: fmt::Debug + Default> fmt::Debug for Cons<H, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}·{:?}", H::default(), T::default())
    }
}

/// Reduce a [`Dimension`] to its canonical sorted [`Term`] list (so that
/// dimensionally-equal types unify). `Dimensionless → Nil`; a base axis inserts
/// its single term; a [`Product`] merges the canonical forms of its children.
pub trait Normalize {
    /// the canonical (sorted, zero-free) association list
    type Output;
}

/// The canonical form of `D` — `<D as Normalize>::Output`.
pub type Normalized<D> = <D as Normalize>::Output;

/// The reciprocal of `D` — `<D as Recip>::Output` (every exponent negated). The
/// type-alias companion to the [`Recip`] trait, so a derived dimension can be
/// written `Product<Num, Reciprocal<DenomDim>>` (used by `Density`, TODO #26).
pub type Reciprocal<D> = <D as Recip>::Output;

impl Normalize for Dimensionless {
    type Output = Nil;
}

impl<D0, D1> Normalize for Product<D0, D1>
where
    D0: Normalize,
    D1: Normalize,
    Normalized<D0>: Merge<Normalized<D1>>,
{
    type Output = <Normalized<D0> as Merge<Normalized<D1>>>::Output;
}

// The canonical forms are *already* normalized, so `Normalize` is idempotent on
// them. This matters for the carrier (Slice 3): a quantity alias is defined as the
// normalized form (e.g. `Quantity<_, Normalized<RadianceDim>, _, _>`), and the
// carrier's `Add` re-normalizes its output; idempotency lands that output back on
// the *same* type so accumulation (`l += l`) type-checks. A `Cons`/`Nil`/`Term`
// only ever arises from `Insert`/`Merge`, which maintain the sorted, zero-free
// invariant, so treating it as its own canonical form is sound.
impl Normalize for Nil {
    type Output = Nil;
}
impl<A: Axis, N: Integer> Normalize for Term<A, N> {
    type Output = Cons<Term<A, N>, Nil>;
}
impl<H, T> Normalize for Cons<H, T> {
    type Output = Cons<H, T>;
}

/// Insert `Term<A, N>` into a sorted list: keep it key-ascending, **sum**
/// exponents on a key collision, and **drop** any term whose exponent cancels to
/// zero. `N` may be zero on entry (it is then dropped).
pub trait Insert<A: Axis, N: Integer> {
    /// the resulting sorted, zero-free list
    type Output;
}

impl<A: Axis, N: Integer> Insert<A, N> for Nil
where
    N: Cmp<Z0>,
    (): ConsHelper<Compare<N, Z0>, A, N, Nil>,
{
    type Output = <() as ConsHelper<Compare<N, Z0>, A, N, Nil>>::Output;
}

impl<A, N, B, M, Tail> Insert<A, N> for Cons<Term<B, M>, Tail>
where
    A: Axis,
    N: Integer,
    B: Axis,
    M: Integer,
    A::Key: Cmp<B::Key>,
    (): InsertHelper<Compare<A::Key, B::Key>, A, N, B, M, Tail>,
{
    type Output = <() as InsertHelper<Compare<A::Key, B::Key>, A, N, B, M, Tail>>::Output;
}

/// Cons `Term<A, N>` onto `Tail`, but drop it when `N == 0`. Keyed on
/// `Ord = Compare<N, Z0>` so the zero case is a distinct, non-overlapping impl.
/// (Public only because it surfaces in [`Insert`]'s associated type; an
/// implementation detail of normalization.)
pub trait ConsHelper<Ord, A: Axis, N: Integer, Tail> {
    /// the list with the term consed (or dropped when zero)
    type Output;
}
impl<A: Axis, N: Integer, Tail> ConsHelper<Equal, A, N, Tail> for () {
    // exponent is zero — drop the term.
    type Output = Tail;
}
impl<A: Axis, N: Integer, Tail> ConsHelper<Less, A, N, Tail> for () {
    type Output = Cons<Term<A, N>, Tail>;
}
impl<A: Axis, N: Integer, Tail> ConsHelper<Greater, A, N, Tail> for () {
    type Output = Cons<Term<A, N>, Tail>;
}

/// Dispatched body of [`Insert`] for a non-empty list, keyed on the key ordering
/// of the incoming axis vs the head axis. (Public only because it surfaces in
/// [`Insert`]'s associated type; an implementation detail of normalization.)
pub trait InsertHelper<Ord, A: Axis, N: Integer, B: Axis, M: Integer, Tail> {
    /// the list after dispatching on the key comparison
    type Output;
}
// incoming key < head key: incoming term sorts first (its `N` is nonzero here —
// zero only enters via a base `Insert`-into-`Nil` or a same-axis cancellation).
impl<A: Axis, N: Integer, B: Axis, M: Integer, Tail> InsertHelper<Less, A, N, B, M, Tail> for () {
    type Output = Cons<Term<A, N>, Cons<Term<B, M>, Tail>>;
}
// same axis: sum exponents, dropping the term if the sum cancels to zero.
impl<A: Axis, N: Integer, B: Axis, M: Integer, Tail> InsertHelper<Equal, A, N, B, M, Tail> for ()
where
    N: Add<M>,
    Sum<N, M>: Integer + Cmp<Z0>,
    (): ConsHelper<Compare<Sum<N, M>, Z0>, A, Sum<N, M>, Tail>,
{
    type Output = <() as ConsHelper<Compare<Sum<N, M>, Z0>, A, Sum<N, M>, Tail>>::Output;
}
// incoming key > head key: keep the head, recurse into the tail.
impl<A: Axis, N: Integer, B: Axis, M: Integer, Tail> InsertHelper<Greater, A, N, B, M, Tail> for ()
where
    Tail: Insert<A, N>,
{
    type Output = Cons<Term<B, M>, <Tail as Insert<A, N>>::Output>;
}

/// Ordered merge of two normalized lists: fold the left list's terms into the
/// right via [`Insert`] (reusing its same-key add / zero-drop).
pub trait Merge<R> {
    /// the merged, still-canonical list
    type Output;
}
impl<R> Merge<R> for Nil {
    type Output = R;
}
impl<B, M, Tail, R> Merge<R> for Cons<Term<B, M>, Tail>
where
    B: Axis,
    M: Integer,
    R: Insert<B, M>,
    Tail: Merge<<R as Insert<B, M>>::Output>,
{
    type Output = <Tail as Merge<<R as Insert<B, M>>::Output>>::Output;
}

/// Two dimensions are equal iff their canonical forms are the same type. Use as
/// a bound — e.g. a carrier's `Add` requires `D2: SameDimension<D1>`.
pub trait SameDimension<Rhs> {}
impl<A, B> SameDimension<B> for A
where
    A: Normalize,
    B: Normalize<Output = Normalized<A>>,
{
}

/// A dimension whose canonical form is empty (`Nil`) — the MC estimator and the
/// `⟨We, L⟩` measurement assert this of their result.
///
/// A non-cancelling dimension is rejected at compile time:
/// ```compile_fail
/// use math::prelude::*;
/// fn assert_dimensionless<D: IsDimensionless>() {}
/// // Ω⁻¹ (a BSDF value) is NOT dimensionless — this fails to compile.
/// assert_dimensionless::<BsdfDim>();
/// ```
pub trait IsDimensionless {}
impl<D> IsDimensionless for D where D: Normalize<Output = Nil> {}

// ---------------------------------------------------------------------------
// Canonical dimensions. Single-axis ones are a base with its exponent;
// multi-axis ones are hand-composed `Product` trees (mirroring how
// `Area`/`ThroughputMeasure` are hand-composed `ProductMeasure`s). The `Dim`
// suffix avoids colliding with the measure/quantity names in the prelude.
// ---------------------------------------------------------------------------

/// `1`
pub type DimensionlessDim = Dimensionless;
/// `L`
pub type LengthDim = Length<P1>;
/// `L²`
pub type AreaDim = Length<P2>;
/// `Φ`
pub type PowerDim = Power<P1>;
/// `Ω`
pub type SolidAngleDim = SolidAngle<P1>;
/// `Λ`
pub type WavelengthDim = Wavelength<P1>;
/// `Ω⁻¹` — a BSDF value `f_s` (per projected solid angle).
pub type BsdfDim = SolidAngle<N1>;
/// `Φ·A⁻¹` — irradiance.
pub type IrradianceDim = Product<Power<P1>, Length<N2>>;
/// `Φ·A⁻¹·Ω⁻¹` — radiance (and, as an adjoint, importance).
pub type RadianceDim = Product<Power<P1>, Product<Length<N2>, SolidAngle<N1>>>;

#[cfg(test)]
mod test {
    use super::*;
    use typenum::Z0;

    /// Compile-time assertion that a type is a [`Dimension`].
    fn assert_dim<D: Dimension>() {}

    #[test]
    fn aliases_are_dimensions() {
        assert_dim::<DimensionlessDim>();
        assert_dim::<LengthDim>();
        assert_dim::<AreaDim>();
        assert_dim::<PowerDim>();
        assert_dim::<SolidAngleDim>();
        assert_dim::<WavelengthDim>();
        assert_dim::<BsdfDim>();
        assert_dim::<IrradianceDim>();
        assert_dim::<RadianceDim>();
    }

    #[test]
    fn same_axis_mul_adds_exponents() {
        // L · L = L²  — nominal match because same-axis exponents merge.
        let _a: AreaDim = LengthDim::default() * LengthDim::default();
        // L² · L = L³
        let _v: Length<typenum::P3> = AreaDim::default() * LengthDim::default();
    }

    #[test]
    fn same_axis_div_subtracts_exponents() {
        // L² / L = L  (nominal)
        let _l: LengthDim = AreaDim::default() / LengthDim::default();
        // L / L = L⁰  (exponent cancels to zero; note: NOT the `Dimensionless`
        // *type* — a Product-normalization pass would be needed for that.)
        let _z: Length<Z0> = LengthDim::default() / LengthDim::default();
    }

    #[test]
    fn dimensionless_is_identity_and_reciprocates() {
        // 1 · Radiance = Radiance  (nominal)
        let _r: RadianceDim = Dimensionless::default() * RadianceDim::default();
        // 1 / Ω = Ω⁻¹ = BsdfDim  (nominal)
        let _b: BsdfDim = Dimensionless::default() / SolidAngleDim::default();
    }

    #[test]
    fn recip_negates_exponent() {
        // Ω⁻¹ then ⁻¹ again = Ω
        fn id<D: Recip>() -> <D as Recip>::Output {
            Default::default()
        }
        let _s: SolidAngle<P1> = id::<BsdfDim>(); // Recip<SolidAngle<N1>> = SolidAngle<P1>
    }

    #[test]
    fn products_nest_on_both_sides() {
        // D0 and D1 of a Product may themselves be Products.
        assert_dim::<Product<AreaDim, RadianceDim>>();
        assert_dim::<Product<RadianceDim, Product<AreaDim, SolidAngleDim>>>();
    }

    #[test]
    fn debug_shows_structure() {
        assert_eq!(format!("{:?}", LengthDim::default()), "Length<1>");
        assert_eq!(format!("{:?}", AreaDim::default()), "Length<2>");
        let r = format!("{:?}", RadianceDim::default());
        assert!(
            r.contains("Power<1>") && r.contains("Length<-2>") && r.contains("SolidAngle<-1>"),
            "got {r}"
        );
    }

    // --- normalization (Slice 1.5) ---------------------------------------

    /// Compile-time assertion that two dimensions are dimensionally equal.
    fn assert_same<A, B>()
    where
        A: SameDimension<B>,
    {
    }
    /// Compile-time assertion that a dimension is dimensionless.
    fn assert_dimensionless<D: IsDimensionless>() {}

    #[test]
    fn normalize_is_order_independent() {
        // Product<A,B> and Product<B,A> reduce to the same canonical list.
        assert_same::<Product<PowerDim, AreaDim>, Product<AreaDim, PowerDim>>();
        assert_same::<
            Product<RadianceDim, AreaDim>,
            Product<AreaDim, RadianceDim>,
        >();
    }

    #[test]
    fn normalize_cancels_to_unit() {
        // Ω · Ω⁻¹ = 1
        assert_dimensionless::<Product<SolidAngleDim, BsdfDim>>();
        // L⁰ = 1 (a same-axis cancellation that lands on `X<Z0>`, not `Nil`,
        // before normalization — normalization drops it).
        assert_dimensionless::<Length<Z0>>();
        assert_dimensionless::<Dimensionless>();
        // L · L⁻¹ via a cross-axis Product also collapses.
        assert_dimensionless::<Product<LengthDim, Length<N1>>>();
    }

    #[test]
    fn normalize_cross_axis_collapse_to_power() {
        // Radiance · A · Ω = Φ  (the rendering-equation power balance).
        assert_same::<
            Product<RadianceDim, Product<AreaDim, SolidAngleDim>>,
            PowerDim,
        >();
        // Irradiance · A = Φ
        assert_same::<Product<IrradianceDim, AreaDim>, PowerDim>();
    }

    #[test]
    fn normalize_retags_are_dimensions() {
        // `Normalized<D>` is itself a `Dimension`, so a carrier can be tagged
        // with the canonical form directly (Slice 3).
        assert_dim::<Normalized<RadianceDim>>();
        assert_dim::<Normalized<Dimensionless>>();
        assert_dim::<Normalized<Length<Z0>>>();
    }

    #[test]
    fn normalized_debug_reads_as_unit_product() {
        // Radiance normalizes to L⁻²·Φ¹·Ω⁻¹ (axis-key ascending: L,Φ,Ω).
        let r = format!("{:?}", Normalized::<RadianceDim>::default());
        assert!(
            r.contains("L^-2") && r.contains("Φ^1") && r.contains("Ω^-1"),
            "got {r}"
        );
        // A fully-cancelled dimension prints as the unit "1".
        assert_eq!(
            format!("{:?}", Normalized::<Product<SolidAngleDim, BsdfDim>>::default()),
            "1"
        );
    }
}
