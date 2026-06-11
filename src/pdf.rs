use std::{
    // marker::PhantomData,
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use crate::prelude::*;

/// A probability density value `dP/dM` — a `T`-valued density taken with respect
/// to the geometric measure `M` (Veach, Radon–Nikodym derivative, Thm. 3.2/3.3).
/// `T` is the (possibly spectral) codomain; `M` is always a scalar geometric
/// measure. The `M` tag is what makes measure-correctness checkable at compile
/// time: see [`Integrand`] / [`Estimate`] for the `f/p` estimator, and
/// [`MeasureConversion`] for changing the reference measure.
//
// `PhantomData<fn() -> M>` (not `*const M`) keeps `PDF: Send + Sync` — important
// for a threaded path tracer — while remaining invariant in `M`.
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct PDF<T: Field, M: Measure> {
    v: T,
    measure: PhantomData<fn() -> M>,
}

impl<T: Field, M: Measure> PDF<T, M> {
    pub fn new(v: T) -> Self {
        Self {
            v,
            measure: PhantomData,
        }
    }
}

// deref, to make things easier. don't need to access pdf.0 anymore, just do *pdf
impl<T: Field, M: Measure> Deref for PDF<T, M> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

impl<T: Field, M: Measure> DerefMut for PDF<T, M> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.v
    }
}

// impl From (and Into) when Measure can be inferred
impl<T: Field, M: Measure> From<T> for PDF<T, M> {
    fn from(v: T) -> Self {
        Self::new(v)
    }
}

/* impl<T: Field, M: Measure> Add for PDF<T, M> {
    type Output = Self;
    // must be under the same field and measure
    fn add(self, rhs: Self) -> Self::Output {
        PDF::new_with_measure(self.v + rhs.v, self.measure.combine(rhs.measure))
    }
} */
impl<T: Field, M: Measure> Mul<T> for PDF<T, M> {
    type Output = Self;
    // must be under the same field and measure
    fn mul(self, rhs: T) -> Self::Output {
        PDF::new(self.v * rhs)
    }
}

/*impl<T: Field, M: Measure> Div for PDF<T, M> {
    // must be under the same field and measure
    // FIXME if you divide two pdfs of the same measure, does that result in a dimensionless quantity? or is it still a pdf? not sure.
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        PDF::new_with_measure(self.v / rhs.v, self.measure.combine(rhs.measure))
    }
}

impl<T: Field, S: Scalar, M: Measure> Mul<S> for PDF<T, M>
where
    T: FromScalar<S>,
{
    type Output = Self;

    fn mul(self, rhs: S) -> Self::Output {
        PDF::new(self.v * T::from_scalar(rhs))
    }
}
impl<T: Field, S: Scalar, M: Measure> Div<S> for PDF<T, M>
where
    T: FromScalar<S>,
{
    type Output = Self;

    fn div(self, rhs: S) -> Self::Output {
        PDF::new(self.v / T::from_scalar(rhs))
    }
} */

// ===========================================================================
// Monte Carlo estimation: integrand / pdf, with the measure checked at the type
// level.
//
// Veach's estimator (eq. 8.8–8.9) is `I_j ≈ f(X) / p(X)`, and it is unbiased
// *only* because `p` is a density with respect to the SAME measure `μ` that the
// integrand `f` is integrated against. We encode that invariant: an
// `Integrand<T, M>` may only be divided by a `PDF<T, M>` with a matching `M`,
// and the result is an `Estimate<T>` whose measure has been integrated out.
// ===========================================================================

/// A value of an integrand `f`, expressed for integration against measure `M`
/// (the `f` in `∫ f dM`). Distinct from [`PDF`], which is a *density* `dP/dM`:
/// an integrand transforms like the measure element, a density transforms
/// inversely. Divide by a matching [`PDF`] to form the Monte Carlo estimate.
///
/// The measure tag `M` is checked at compile time: an `Integrand<T, M>` can only
/// be divided by a `PDF<T, M>` with the *same* `M`. This is the type-level form
/// of Veach's unbiasedness condition (eq. 8.9) — `f(X)/p(X)` is only a valid
/// estimator when `f` and `p` reference the same measure. Mismatches below are
/// rejected by the compiler, so there is no runtime check to test for: the
/// `compile_fail` examples assert that the bad code does not build.
///
/// Matching measures divide fine and yield a measure-free [`Estimate`]:
///
/// ```
/// use math::prelude::*;
///
/// let f: Integrand<f32, SolidAngle> = Integrand::new(6.0);
/// let p: PDF<f32, SolidAngle> = PDF::new(2.0);
/// let est: Estimate<f32> = f / p; // measures match → OK
/// assert_eq!(*est, 3.0);
/// ```
///
/// An area integrand cannot be divided by a solid-angle density:
///
/// ```compile_fail
/// use math::prelude::*;
///
/// let f: Integrand<f32, Area> = Integrand::new(6.0);
/// let p: PDF<f32, SolidAngle> = PDF::new(2.0);
/// let _est = f / p; // ERROR: no `Div` impl — Area ≠ SolidAngle
/// ```
///
/// Nor can a throughput integrand be divided by an area density:
///
/// ```compile_fail
/// use math::prelude::*;
///
/// let f: Integrand<f32, Throughput> = Integrand::new(1.0);
/// let p: PDF<f32, Area> = PDF::new(1.0);
/// let _est = f / p; // ERROR: Throughput ≠ Area
/// ```
///
/// [`PDF::convert`] changes the measure tag, so a converted density no longer
/// matches an integrand taken against the original measure:
///
/// ```compile_fail
/// use math::prelude::*;
///
/// let f: Integrand<f32, SolidAngle> = Integrand::new(1.0);
/// let p: PDF<f32, SolidAngle> = PDF::new(1.0);
/// // converting to projected solid angle re-tags the density:
/// let p_psa: PDF<f32, ProjectedSolidAngle> =
///     p.convert(DirectionalGeom { cos_theta: 0.5 });
/// let _est = f / p_psa; // ERROR: SolidAngle ≠ ProjectedSolidAngle
/// ```
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct Integrand<T: Field, M: Measure> {
    v: T,
    measure: PhantomData<fn() -> M>,
}

impl<T: Field, M: Measure> Integrand<T, M> {
    #[inline(always)]
    pub fn new(v: T) -> Self {
        Self {
            v,
            measure: PhantomData,
        }
    }
}

impl<T: Field, M: Measure> Deref for Integrand<T, M> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

impl<T: Field, M: Measure> From<T> for Integrand<T, M> {
    #[inline(always)]
    fn from(v: T) -> Self {
        Self::new(v)
    }
}

/// The result of a Monte Carlo estimate `f(X) / p(X)` (Veach §8.2). The
/// integration measure has cancelled, so this value carries no measure tag — it
/// estimates the measurement `I_j`. This is what a path tracer accumulates into
/// the framebuffer.
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct Estimate<T: Field> {
    v: T,
}

impl<T: Field> Estimate<T> {
    #[inline(always)]
    pub fn new(v: T) -> Self {
        Self { v }
    }
}

impl<T: Field> Deref for Estimate<T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

/// `f(X) / p(X)` — the measures must match (same `M`) or this will not compile.
impl<T: Field, M: Measure> Div<PDF<T, M>> for Integrand<T, M> {
    type Output = Estimate<T>;
    #[inline(always)]
    fn div(self, pdf: PDF<T, M>) -> Estimate<T> {
        Estimate::new(self.v / *pdf)
    }
}

impl<T: Field> Add for Estimate<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Estimate::new(self.v + rhs.v)
    }
}

impl<T: Field> AddAssign for Estimate<T> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.v += rhs.v;
    }
}

/// Scale an estimate by a dimensionless weight (e.g. a MIS weight, or 1/N).
impl<T: Field> Mul<T> for Estimate<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: T) -> Self {
        Estimate::new(self.v * rhs)
    }
}

// ===========================================================================
// Measure conversions as a single Radon–Nikodym chain rule.
//
// A density transforms between measures `From` and `To` by the Radon–Nikodym
// derivative `dFrom/dTo`:  p_To = p_From · (dFrom/dTo)   (chain rule for 8.9).
// `MeasureConversion::<From, To>::jacobian()` returns exactly `dFrom/dTo`, and
// [`PDF::convert`] multiplies by it. Inverses are the reciprocal, and chains
// (e.g. Area→SolidAngle→ProjectedSolidAngle) compose by `.convert().convert()`.
//
// NOTE: directions here follow Veach eq. (8.10) `p_area = p_σ · cos/r²` and
// (8.11) `p_σ⊥ = p_σ · 1/cos`. The legacy `convert_to_*` methods below now
// delegate to `convert`, but are deprecated: their original bodies multiplied by
// the reciprocal factors (transforming the value like a measure element, not a
// density), so the delegated result flips relative to the old one. Prefer
// `convert` everywhere.
// ===========================================================================

/// Radon–Nikodym derivative `dFrom/dTo` between two measures on the same domain,
/// carried by a value holding the geometry needed to evaluate it.
pub trait MeasureConversion<From: Measure, To: Measure> {
    /// `dFrom/dTo` at the conversion point.
    fn jacobian(&self) -> f32;
}

impl<T: Field + FromScalar<f32>, From: Measure> PDF<T, From> {
    /// Re-express this density with respect to a different measure `To`,
    /// multiplying by the Radon–Nikodym derivative `dFrom/dTo`.
    #[inline(always)]
    pub fn convert<To: Measure, C: MeasureConversion<From, To>>(self, conv: C) -> PDF<T, To> {
        PDF::new(self.v * T::from_scalar(conv.jacobian()))
    }
}

/// `cos θ = |ω · N|` at a vertex — the geometry for converting a directional
/// density between ordinary and projected solid angle (Veach eq. 3.16 / 8.11).
#[derive(Copy, Clone, Debug)]
pub struct DirectionalGeom {
    pub cos_theta: f32,
}

/// Geometry relating a sampled surface point to the vertex it is seen from:
/// `cos_theta = |ω · N(x')|` at the sampled point and `dist_sq = ‖x − x'‖²`
/// (Veach eq. 8.10).
#[derive(Copy, Clone, Debug)]
pub struct AreaGeom {
    pub cos_theta: f32,
    pub dist_sq: f32,
}

// dσ/dσ⊥ = 1/|cos θ|   (Veach 8.11)
impl MeasureConversion<SolidAngle, ProjectedSolidAngle> for DirectionalGeom {
    #[inline(always)]
    fn jacobian(&self) -> f32 {
        self.cos_theta.abs().recip()
    }
}

// dσ⊥/dσ = |cos θ|
impl MeasureConversion<ProjectedSolidAngle, SolidAngle> for DirectionalGeom {
    #[inline(always)]
    fn jacobian(&self) -> f32 {
        self.cos_theta.abs()
    }
}

// dσ/dA = |cos θ| / r²   (Veach 8.10):  p_area = p_σ · cos/r²
impl MeasureConversion<SolidAngle, Area> for AreaGeom {
    #[inline(always)]
    fn jacobian(&self) -> f32 {
        self.cos_theta.abs() / self.dist_sq
    }
}

// dA/dσ = r² / |cos θ|
impl MeasureConversion<Area, SolidAngle> for AreaGeom {
    #[inline(always)]
    fn jacobian(&self) -> f32 {
        self.dist_sq / self.cos_theta.abs()
    }
}

/// Geometry of a full path edge between two surface points: `cos_i`/`cos_o` are
/// the cosines at the two endpoints and `dist_sq = ‖x − x'‖²`. Converts an area
/// density directly to a projected-solid-angle density, composing the A↔σ and
/// σ↔σ⊥ steps (Veach eq. 8.2: the geometric term `G = |cos_i · cos_o| / r²`).
/// Equivalent to chaining [`AreaGeom`] then [`DirectionalGeom`] through a
/// [`SolidAngle`] intermediate.
#[derive(Copy, Clone, Debug)]
pub struct EdgeGeom {
    pub cos_i: f32,
    pub cos_o: f32,
    pub dist_sq: f32,
}

// dA/dσ⊥ = r² / (|cos_i| |cos_o|) = 1/G   (Veach eq. 8.2)
impl MeasureConversion<Area, ProjectedSolidAngle> for EdgeGeom {
    #[inline(always)]
    fn jacobian(&self) -> f32 {
        self.dist_sq / (self.cos_i.abs() * self.cos_o.abs())
    }
}

// ---------------------------------------------------------------------------
// Legacy explicit conversions. These now delegate to `convert` /
// `MeasureConversion` so there is a single source of truth for the Jacobians,
// but they are DEPRECATED: their original hand-written bodies multiplied by the
// *reciprocal* of the Radon–Nikodym density factor (they scaled the value like a
// measure element, not a density), so routing them through `convert` flips the
// numeric result to the Veach-correct direction. Any caller (e.g. the path
// tracer) that silently relied on the old behavior — or carried a compensating
// reciprocal elsewhere — will now produce different values. Migrate to `convert`
// and verify against the tracer. See MEASURE_TYPE_TODOS.md #2.
// ---------------------------------------------------------------------------

// special conversions
impl<T: Field> PDF<T, SolidAngle> {
    #[deprecated(
        note = "POSSIBLE BUG: the old body multiplied by |cos θ| (the measure-element \
                factor dσ⊥/dσ), but a density converts by dσ/dσ⊥ = 1/|cos θ| (Veach \
                eq. 8.11). This now delegates to `convert` and returns the reciprocal \
                of the old result — verify the path tracer didn't depend on the old \
                direction. Use `pdf.convert(DirectionalGeom { cos_theta })` instead."
    )]
    pub fn convert_to_projected_solid_angle<S: Scalar + ToScalar<f32>>(
        &self,
        cos_theta: S,
    ) -> PDF<T, ProjectedSolidAngle>
    where
        T: FromScalar<f32>,
    {
        (*self).convert(DirectionalGeom {
            cos_theta: cos_theta.to_scalar(),
        })
    }
}

impl<T: Field> PDF<T, Area> {
    #[deprecated(
        note = "POSSIBLE BUG: the old body multiplied by |cos θ| / r² (the measure-element \
                factor dσ/dA), but a density converts by dA/dσ = r² / |cos θ| (Veach \
                eq. 8.10). This now delegates to `convert` and returns the reciprocal \
                of the old result — verify the path tracer didn't depend on the old \
                direction. Use `pdf.convert(AreaGeom { cos_theta, dist_sq })` instead."
    )]
    pub fn convert_to_solid_angle<S: Scalar + ToScalar<f32>>(
        &self,
        cos_theta: S,
        distance_squared: S,
    ) -> PDF<T, SolidAngle>
    where
        T: FromScalar<f32>,
    {
        (*self).convert(AreaGeom {
            cos_theta: cos_theta.to_scalar(),
            dist_sq: distance_squared.to_scalar(),
        })
    }
}

impl<T: Field> PDF<T, Area> {
    #[deprecated(
        note = "POSSIBLE BUG: the old body multiplied by |cos_i · cos_o| / r² (the geometric \
                term G = dσ⊥/dA), but a density converts by dA/dσ⊥ = r² / |cos_i · cos_o| \
                (Veach eq. 8.2). This now delegates to `convert` and returns the reciprocal \
                of the old result — verify the path tracer didn't depend on the old \
                direction. Use `pdf.convert(EdgeGeom { cos_i, cos_o, dist_sq })` instead."
    )]
    pub fn convert_to_projected_solid_angle<S: Scalar + ToScalar<f32>>(
        &self,
        cos_i: S,
        cos_o: S,
        distance_squared: S,
    ) -> PDF<T, ProjectedSolidAngle>
    where
        T: FromScalar<f32>,
    {
        (*self).convert(EdgeGeom {
            cos_i: cos_i.to_scalar(),
            cos_o: cos_o.to_scalar(),
            dist_sq: distance_squared.to_scalar(),
        })
    }
}

// // impl<T> PDF<T, ProjectedSolidAngle> where T: Field {}
// impl<T: Field> PDF<T, ProjectedSolidAngle> {
//     fn convert_to_throughput(self, area_pdf: PDF<T, Area>) -> PDF<T, Throughput> {
//         (*area_pdf * *self).into()
//     }
// }

#[cfg(test)]
mod test {
    use super::*;
    use crate::spaces::{DirectionalSector, SphericalCoordinates};

    type SA = SolidAngle;

    /// Deterministic stratified midpoint grid over `[0,1)²`: `N×N` samples at the
    /// stratum centers. No RNG, so the Monte Carlo tests below are reproducible.
    fn stratified_grid(n: usize) -> impl Iterator<Item = (f32, f32)> {
        (0..n).flat_map(move |i| {
            (0..n).map(move |j| {
                let u = (i as f32 + 0.5) / n as f32;
                let v = (j as f32 + 0.5) / n as f32;
                (u, v)
            })
        })
    }

    #[test]
    fn estimator_cancels_measure() {
        // f(X)/p(X): integrand and pdf must share the measure; result is measure-free.
        let f: Integrand<f32, Area> = Integrand::new(6.0);
        let p: PDF<f32, Area> = PDF::new(2.0);
        let est: Estimate<f32> = f / p;
        assert_eq!(*est, 3.0);
    }

    #[test]
    fn solid_angle_is_chart_independent() {
        // The whole point of TODO #4: `SolidAngle` is one measure over the
        // `Directions` domain, no longer parameterized by the chart. A density
        // whose value came from the spherical-coordinate Jacobian and one from
        // the cone (DirectionalSector) chart now have the SAME type, so either
        // can divide an `Integrand<_, SolidAngle>`. (Before the split these were
        // distinct types `SolidAngle<SphericalCoordinates>` vs
        // `SolidAngle<DirectionalSector>` and this would not compile.)
        let theta = std::f32::consts::FRAC_PI_3;
        let d_spherical =
            <SolidAngle as ChartedMeasure<SphericalCoordinates>>::differential_measure((0.0, theta));
        let d_cone = <SolidAngle as ChartedMeasure<DirectionalSector>>::differential_measure([
            0.0, 0.0, 1.0,
        ]);

        let p_from_spherical: PDF<f32, SolidAngle> = PDF::new(d_spherical);
        let p_from_cone: PDF<f32, SolidAngle> = PDF::new(d_cone);

        // both share the measure tag, so both divide an integrand of the same tag
        let _e1: Estimate<f32> = Integrand::<f32, SolidAngle>::new(1.0) / p_from_spherical;
        let _e2: Estimate<f32> = Integrand::<f32, SolidAngle>::new(1.0) / p_from_cone;
        assert_eq!(d_cone, 1.0); // cone chart Jacobian is unity
    }

    #[test]
    fn convert_solid_angle_to_area_matches_veach() {
        // Veach eq. 8.10: p_area = p_σ · cos/r²
        let p_sa: PDF<f32, SA> = PDF::new(0.5);
        let geom = AreaGeom {
            cos_theta: 0.5,
            dist_sq: 4.0,
        };
        let p_area: PDF<f32, Area> = p_sa.convert(geom);
        assert!((*p_area - 0.5 * 0.5 / 4.0).abs() < 1e-6);
    }

    #[test]
    fn convert_round_trips() {
        // Area → SolidAngle → Area returns the original density.
        let p_area: PDF<f32, Area> = PDF::new(0.75);
        let geom = AreaGeom {
            cos_theta: 0.3,
            dist_sq: 2.5,
        };
        let p_sa: PDF<f32, SA> = p_area.convert(geom);
        let back: PDF<f32, Area> = p_sa.convert(geom);
        assert!((*back - 0.75).abs() < 1e-6);
    }

    #[test]
    fn convert_solid_angle_to_projected_matches_veach() {
        // Veach eq. 8.11: p_σ⊥ = p_σ · 1/cos
        let p_sa: PDF<f32, SA> = PDF::new(1.0);
        let p_psa: PDF<f32, ProjectedSolidAngle> =
            p_sa.convert(DirectionalGeom { cos_theta: 0.25 });
        assert!((*p_psa - 4.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Monte Carlo change-of-variables tests.
    //
    // For X drawn from a pdf `p` w.r.t. measure M, the estimator of the constant
    // integrand 1 is  E[1/p(X)] = ∫ (1/p) p dM = ∫_supp dM = (M-measure of the
    // support). Converting `p` to a different measure M' (via `convert`) changes
    // that constant to the M'-measure of the *same* support — and to a value we
    // can check against closed form. A conversion in the wrong (reciprocal)
    // direction would diverge or land on a different constant, so these tests
    // pin the *direction* of the Radon–Nikodym factor, not just its magnitude.
    // -----------------------------------------------------------------------

    /// Uniform-solid-angle sampling of the upper hemisphere: cosθ = u₁, p_σ =
    /// 1/(2π). Returns (cos_theta, p_sigma) per sample. The integrand-1 estimate
    /// E[1/p_σ] is the solid-angle measure of the hemisphere = 2π.
    #[test]
    fn mc_hemisphere_solid_angle_measure_is_2pi() {
        const N: usize = 64;
        let mut acc = 0.0f32;
        let mut count = 0u32;
        for (_u1, _u2) in stratified_grid(N) {
            // uniform over the hemisphere in solid angle
            let p_sigma: PDF<f32, SA> = PDF::new(1.0 / (2.0 * std::f32::consts::PI));
            let est: Estimate<f32> = Integrand::<f32, SA>::new(1.0) / p_sigma;
            acc += *est;
            count += 1;
        }
        let mean = acc / count as f32;
        // every sample yields exactly 2π, so this is essentially exact.
        assert!(
            (mean - 2.0 * std::f32::consts::PI).abs() < 1e-4,
            "got {mean}, want 2π"
        );
    }

    /// Same uniform-solid-angle samples, but convert the pdf to projected solid
    /// angle (Veach eq. 8.11, p_σ⊥ = p_σ/cosθ). Now E[1/p_σ⊥] is the *projected*
    /// solid angle of the hemisphere = π (the area of the unit disk). The buggy
    /// reciprocal direction (p_σ⊥ = p_σ·cosθ) would make 1/p_σ⊥ = 2π/cosθ, whose
    /// expectation diverges — so this test fails loudly if the direction flips.
    #[test]
    fn mc_convert_to_projected_gives_disk_area_pi() {
        const N: usize = 64;
        let mut acc = 0.0f32;
        let mut count = 0u32;
        for (u1, _u2) in stratified_grid(N) {
            let cos_theta = u1; // cosθ ~ U(0,1) for uniform-solid-angle sampling
            let p_sigma: PDF<f32, SA> = PDF::new(1.0 / (2.0 * std::f32::consts::PI));
            let p_proj: PDF<f32, ProjectedSolidAngle> =
                p_sigma.convert(DirectionalGeom { cos_theta });
            let est: Estimate<f32> = Integrand::<f32, ProjectedSolidAngle>::new(1.0) / p_proj;
            acc += *est;
            count += 1;
        }
        let mean = acc / count as f32;
        // 1/p_σ⊥ = 2π·cosθ is linear in u₁, so the midpoint grid integrates it
        // exactly: mean = 2π·E[cosθ] = 2π·½ = π.
        assert!(
            (mean - std::f32::consts::PI).abs() < 1e-3,
            "got {mean}, want π"
        );
    }

    /// Area→solid-angle conversion checked against the analytic solid angle
    /// subtended by a square. A square light of side `L` sits at height `h`
    /// directly above the origin, facing down. Sampling its surface uniformly
    /// (p_A = 1/L²) and converting to solid angle (Veach eq. 8.10), the
    /// integrand-1 estimate E[1/p_σ] converges to the subtended solid angle
    /// Ω = 4·atan(ab / (h·√(a²+b²+h²))) with a = b = L/2.
    #[test]
    fn mc_convert_area_to_solid_angle_matches_subtended_angle() {
        const N: usize = 64;
        let l = 2.0f32;
        let h = 1.0f32;
        let a = l / 2.0;
        let omega = 4.0 * (a * a / (h * (a * a + a * a + h * h).sqrt())).atan();

        let p_area: PDF<f32, Area> = PDF::new(1.0 / (l * l));
        let mut acc = 0.0f32;
        let mut count = 0u32;
        for (u, v) in stratified_grid(N) {
            // map [0,1)² onto the square [-L/2, L/2]²
            let x = -a + l * u;
            let y = -a + l * v;
            let dist_sq = x * x + y * y + h * h;
            let cos_area = h / dist_sq.sqrt(); // |N'·ω| at the light surface
            let p_sigma: PDF<f32, SA> = p_area.convert(AreaGeom {
                cos_theta: cos_area,
                dist_sq,
            });
            let est: Estimate<f32> = Integrand::<f32, SA>::new(1.0) / p_sigma;
            acc += *est;
            count += 1;
        }
        let mean = acc / count as f32;
        assert!(
            (mean - omega).abs() < 1e-2,
            "got {mean}, want subtended solid angle {omega}"
        );
    }

    /// `EdgeGeom` (direct Area→ProjectedSolidAngle, Veach eq. 8.2) must agree with
    /// composing AreaGeom (A→σ) then DirectionalGeom (σ→σ⊥). Verified under the
    /// same Monte Carlo samples as above: both estimators of the integrand 1 must
    /// converge to the same projected solid angle subtended by the square.
    #[test]
    fn mc_edge_geom_matches_area_then_directional_composition() {
        const N: usize = 64;
        let l = 2.0f32;
        let h = 1.0f32;
        let a = l / 2.0;

        let p_area: PDF<f32, Area> = PDF::new(1.0 / (l * l));
        let mut acc_edge = 0.0f32;
        let mut acc_chain = 0.0f32;
        let mut count = 0u32;
        for (u, v) in stratified_grid(N) {
            let x = -a + l * u;
            let y = -a + l * v;
            let dist_sq = x * x + y * y + h * h;
            let r = dist_sq.sqrt();
            // both endpoints' planes are parallel here, so cos_i == cos_o == h/r
            let cos = h / r;

            let p_edge: PDF<f32, ProjectedSolidAngle> = p_area.convert(EdgeGeom {
                cos_i: cos,
                cos_o: cos,
                dist_sq,
            });
            let p_chain: PDF<f32, ProjectedSolidAngle> = p_area
                .convert::<SA, _>(AreaGeom {
                    cos_theta: cos,
                    dist_sq,
                })
                .convert(DirectionalGeom { cos_theta: cos });

            acc_edge += *(Integrand::<f32, ProjectedSolidAngle>::new(1.0) / p_edge);
            acc_chain += *(Integrand::<f32, ProjectedSolidAngle>::new(1.0) / p_chain);
            count += 1;
        }
        let mean_edge = acc_edge / count as f32;
        let mean_chain = acc_chain / count as f32;
        assert!(
            (mean_edge - mean_chain).abs() < 1e-3,
            "EdgeGeom ({mean_edge}) disagrees with AreaGeom∘DirectionalGeom ({mean_chain})"
        );
    }
}
