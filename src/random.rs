use crate::prelude::*;
use thermite::simd::Simd;

#[inline(always)]
pub fn debug_random() -> f32 {
    rand::random()
}

// ===========================================================================
// Generic warp cores (single source of truth for sample + pdf).
//
// Each `*_core` is the change-of-variables `T : [0,1)ⁿ → ℝ³` written once
// against `SampleField`. Instantiating it with `f32` produces the sample; with
// `Dual<2>` it additionally carries the Jacobian, from which the pdf is the
// reciprocal Gram determinant `1/√det(JᵀJ)` (= the density w.r.t. the surface
// measure induced on the warp's image). See `dual.rs` and the research plan.
// ===========================================================================

/// Equal-area cylindrical map onto the unit sphere. pdf = 1/(4π) wrt solid angle.
#[inline(always)]
fn on_unit_sphere_core<F: SampleField>(x: F, y: F) -> [F; 3] {
    let phi = x * F::constant(2.0 * PI);
    let z = y * F::constant(2.0) - F::constant(1.0);
    let r = (F::constant(1.0) - z * z).sqrt();
    let (s, c) = phi.sin_cos();
    [r * c, r * s, z]
}

/// Concentric-free uniform disk map (`z` held at 0). pdf = 1/π wrt area.
#[inline(always)]
fn in_unit_disk_core<F: SampleField>(x: F, y: F) -> [F; 3] {
    let u = x * F::constant(2.0 * PI);
    let v = y.sqrt();
    let (s, c) = u.sin_cos();
    [c * v, s * v, F::constant(0.0)]
}

/// Cosine-weighted hemisphere map. pdf = cosθ/π wrt solid angle (1/π wrt
/// projected solid angle — reach it with `pdf.convert(DirectionalGeom{..})`).
#[inline(always)]
fn cosine_direction_core<F: SampleField>(u: F, v: F) -> [F; 3] {
    let z = (F::constant(1.0) - v).sqrt();
    let phi = u * F::constant(2.0 * PI);
    let (s, c) = phi.sin_cos();
    let vsqrt = v.sqrt();
    [c * vsqrt, s * vsqrt, z]
}

/// Spherical-coordinate map onto the unit ball. pdf = 3/(4π) wrt volume.
#[inline(always)]
fn in_unit_sphere_core<F: SampleField>(x: F, y: F, z: F) -> [F; 3] {
    let u = x * F::constant(2.0 * PI);
    let v = (y * F::constant(2.0) - F::constant(1.0)).acos();
    let w = z.powf(1.0 / 3.0);
    let (su, cu) = u.sin_cos();
    let (sv, cv) = v.sin_cos();
    [cu * sv * w, cv * w, su * sv * w]
}

/// Uniform map onto the spherical cap of half-angle `acos(√(1-k))` seen from a
/// point, where `k = radius²/distance²`. pdf is uniform over the cap's solid
/// angle. The cap geometry `k` is a constant w.r.t. the random inputs, so it
/// enters as a [`SampleField::constant`].
#[inline(always)]
fn to_sphere_core<F: SampleField>(r1: F, r2: F, k: f32) -> [F; 3] {
    let cos_theta_max = (F::constant(1.0) - F::constant(k)).sqrt();
    let z = F::constant(1.0) + r2 * (cos_theta_max - F::constant(1.0));
    let phi = r1 * F::constant(2.0 * PI);
    let (s, c) = phi.sin_cos();
    let sqrt_1_z2 = (F::constant(1.0) - z * z).sqrt();
    [c * sqrt_1_z2, s * sqrt_1_z2, z]
}

/// Power-cosine (Phong) lobe around +z, concentration exponent `n ≥ 0`.
/// `cosθ = u₁^{1/(n+1)}`, `φ = 2π·u₂`. pdf = (n+1)/(2π)·cosⁿθ wrt solid angle.
/// `n=0` → uniform hemisphere; `n=1` → the cosine lobe; `n→∞` → tight around +z.
/// Built in spherical form so the result is unit-length by construction (no
/// `.normalized()`), unlike the deprecated [`weighted_cosine_direction`].
#[inline(always)]
fn power_cosine_core<F: SampleField>(u1: F, u2: F, n: f32) -> [F; 3] {
    let cos_t = u1.powf(1.0 / (n + 1.0));
    let sin_t = (F::constant(1.0) - cos_t * cos_t).sqrt();
    let phi = u2 * F::constant(2.0 * PI);
    let (s, c) = phi.sin_cos();
    [c * sin_t, s * sin_t, cos_t]
}

/// GGX / Trowbridge-Reitz microfacet lobe (micronormal) around +z, roughness
/// `α ∈ (0,1]`. `cosθ = √((1−u₁)/(1+(α²−1)·u₁))`, `φ = 2π·u₂` (Walter et al.
/// 2007 NDF importance sampling). pdf = D(θ)·cosθ wrt solid angle, with
/// `D = α²/(π·((α²−1)cos²θ+1)²)`. Small `α` → tight (mirror-like); large `α` →
/// broad. Built in spherical form so the result is unit-length by construction.
#[inline(always)]
fn ggx_core<F: SampleField>(u1: F, u2: F, alpha: f32) -> [F; 3] {
    let a2m1 = F::constant(alpha * alpha - 1.0);
    let cos_t = ((F::constant(1.0) - u1) / (F::constant(1.0) + a2m1 * u1)).sqrt();
    let sin_t = (F::constant(1.0) - cos_t * cos_t).sqrt();
    let phi = u2 * F::constant(2.0 * PI);
    let (s, c) = phi.sin_cos();
    [c * sin_t, s * sin_t, cos_t]
}

#[inline(always)]
fn vec3_of<S: Simd>(c: [f32; 3]) -> Vec3<S> {
    Vec3::new(c[0], c[1], c[2])
}

/// Nudge a raw `[0, 1)` sample coordinate into the open interval `(0, 1)`.
///
/// The RNG emits the half-open `[0, 1)`, so `0.0` is a real output. Several warp
/// cores contain a `√·` (or `acos`, `xᵖ` with `p < 1`) whose derivative is `+∞`
/// when its argument hits `0` — at a sphere/disk pole or a lobe axis. There the
/// autodiff Jacobian has one `∞` column meeting a zero-length column, so the
/// Gram-determinant pdf `1/√det(JᵀJ)` forms `0·∞ = NaN` (this NaN'd a
/// `sharp_light` render). The warp pdfs are continuous, so evaluating a hair off
/// the endpoint returns the correct finite density — exact for the azimuthally
/// symmetric warps, whose orthogonal Jacobian columns make `det = g00·g11` an
/// ε-independent product (no cancellation). Applied identically on the value and
/// pdf paths so the produced sample is unchanged between them. See the
/// `boundary` tests.
#[inline(always)]
fn open_unit(t: f32) -> f32 {
    // One ulp below 1.0 is already in contract; only the closed `0.0` endpoint
    // (and its `1.0` mirror, unreachable here) is singular. EPS is well above the
    // subnormal range so the nudged `√·` arguments stay comfortably normalized.
    const EPS: f32 = 1e-6;
    t.clamp(EPS, 1.0 - EPS)
}

/// Run a 2-input warp core on dual inputs and split into (sample, pdf-value),
/// where the pdf value is the reciprocal Gram determinant of the warp Jacobian.
#[inline(always)]
fn warp_with_pdf<S: Simd>(
    core: impl Fn(Dual<2>, Dual<2>) -> [Dual<2>; 3],
    r: Sample2D,
) -> (Vec3<S>, f32) {
    let out = core(
        Dual::variable(open_unit(r.x), 0),
        Dual::variable(open_unit(r.y), 1),
    );
    let v = vec3_of([out[0].value(), out[1].value(), out[2].value()]);
    (v, reciprocal_gram_det_2(&out))
}

/// As [`warp_with_pdf`] but for a three-input (full-dimensional) warp; the pdf
/// value is the reciprocal `3×3` Jacobian determinant.
#[inline(always)]
fn warp_with_pdf_3<S: Simd>(
    core: impl Fn(Dual<3>, Dual<3>, Dual<3>) -> [Dual<3>; 3],
    r: Sample3D,
) -> (Vec3<S>, f32) {
    let out = core(
        Dual::variable(open_unit(r.x), 0),
        Dual::variable(open_unit(r.y), 1),
        Dual::variable(open_unit(r.z), 2),
    );
    let v = vec3_of([out[0].value(), out[1].value(), out[2].value()]);
    (v, reciprocal_det_3(&out))
}

/// Uniformly distributed wrt the volume measure on the unit ball.
#[inline(always)]
pub fn random_in_unit_sphere<S: Simd>(r: Sample3D) -> Vec3<S> {
    vec3_of(
        in_unit_sphere_core::<f32>(open_unit(r.x), open_unit(r.y), open_unit(r.z))
            .map(SampleField::value),
    )
}

/// As [`random_in_unit_sphere`], but also returns the (uniform) volume pdf
/// (`3/(4π)`) computed automatically from the warp Jacobian.
#[inline(always)]
pub fn random_in_unit_sphere_with_pdf<S: Simd>(r: Sample3D) -> (Vec3<S>, ScalarPDF<Volume>) {
    let (v, p) = warp_with_pdf_3::<S>(in_unit_sphere_core, r);
    (v, PDF::new(Vector::<f32>::splat(p)))
}

/// Uniformly distributed wrt the surface area / solid angle measure.
#[inline(always)]
pub fn random_on_unit_sphere<S: Simd>(r: Sample2D) -> Vec3<S> {
    vec3_of(on_unit_sphere_core::<f32>(open_unit(r.x), open_unit(r.y)).map(SampleField::value))
}

/// As [`random_on_unit_sphere`], but also returns the (uniform) solid-angle pdf.
/// changing away from using dual numbers for this one as it's analytically available and prone to pole singularities
#[inline(always)]
pub fn random_on_unit_sphere_with_pdf<S: Simd>(r: Sample2D) -> (Vec3<S>, ScalarPDF<SolidAngle>) {
    // let (v, p) = warp_with_pdf::<S>(on_unit_sphere_core, r);
    // debug_assert!(!p.is_nan(), "{r:?}");
    let v = random_on_unit_sphere(r);
    // (v, ScalarPDF::new_from(p))
    (v, PDF::new(Vector::<f32>::splat((4.0 * PI).recip())))
}

/// Uniformly distributed wrt the area measure on the unit disk.
#[inline(always)]
pub fn random_in_unit_disk<S: Simd>(r: Sample2D) -> Vec3<S> {
    vec3_of(in_unit_disk_core::<f32>(open_unit(r.x), open_unit(r.y)).map(SampleField::value))
}

/// As [`random_in_unit_disk`], but also returns the (uniform) area pdf computed
/// automatically from the warp Jacobian.
#[inline(always)]
pub fn random_in_unit_disk_with_pdf<S: Simd>(r: Sample2D) -> (Vec3<S>, ScalarPDF<Area>) {
    let (v, p) = warp_with_pdf::<S>(in_unit_disk_core, r);
    (v, PDF::new(Vector::<f32>::splat(p)))
}

/// Cosine-weighted hemisphere direction. Uniform wrt projected solid angle.
#[inline(always)]
pub fn random_cosine_direction<S: Simd>(r: Sample2D) -> Vec3<S> {
    vec3_of(cosine_direction_core::<f32>(open_unit(r.x), open_unit(r.y)).map(SampleField::value))
}

/// As [`random_cosine_direction`], but also returns the solid-angle pdf
/// (`cosθ/π`) computed automatically from the warp Jacobian. Convert to a
/// projected-solid-angle pdf (`1/π`) with `pdf.convert(DirectionalGeom { .. })`.
#[inline(always)]
pub fn random_cosine_direction_with_pdf<S: Simd>(r: Sample2D) -> (Vec3<S>, ScalarPDF<SolidAngle>) {
    let (v, p) = warp_with_pdf::<S>(cosine_direction_core, r);
    (v, PDF::new(Vector::<f32>::splat(p)))
}

/// Power-cosine (Phong) lobe direction around +z, concentration exponent `n`.
/// Orient to an arbitrary central direction with
/// `TangentFrame::from_normal(d).to_world(..)`.
#[inline(always)]
pub fn power_cosine_direction<S: Simd>(r: Sample2D, n: f32) -> Vec3<S> {
    vec3_of(power_cosine_core::<f32>(open_unit(r.x), open_unit(r.y), n).map(SampleField::value))
}

/// As [`power_cosine_direction`], but also returns the solid-angle pdf
/// `(n+1)/(2π)·cosⁿθ`.
///
/// Uses the closed form rather than the autodiff Gram determinant: at the lobe
/// axis f32 rounds `cosθ` to exactly `1.0`, so `sinθ = √(1−cos²θ) = 0` and the
/// autodiff Jacobian forms `0·∞ = NaN` there — an input-nudge can't escape the
/// rounding for high `n`. The analytic form is finite everywhere (`boundary`
/// tests) and is validated against an independent re-derivation of the closed
/// form on the interior by `power_cosine_pdf_matches_closed_form`.
#[inline(always)]
pub fn power_cosine_direction_with_pdf<S: Simd>(
    r: Sample2D,
    n: f32,
) -> (Vec3<S>, ScalarPDF<SolidAngle>) {
    let v = power_cosine_direction::<S>(r, n);
    let cos_theta = v.z().max(0.0); // lobe is around +z
    let p = (n + 1.0) / (2.0 * PI) * cos_theta.powf(n);
    (v, PDF::new(Vector::<f32>::splat(p)))
}

/// GGX / Trowbridge-Reitz microfacet lobe direction around +z, roughness `alpha`.
/// Orient to an arbitrary central direction with
/// `TangentFrame::from_normal(d).to_world(..)`.
#[inline(always)]
pub fn ggx_direction<S: Simd>(r: Sample2D, alpha: f32) -> Vec3<S> {
    vec3_of(ggx_core::<f32>(open_unit(r.x), open_unit(r.y), alpha).map(SampleField::value))
}

/// As [`ggx_direction`], but also returns the solid-angle pdf `D(θ)·cosθ`, with
/// `D = α²/(π·((α²−1)cos²θ+1)²)`.
///
/// Uses the closed form rather than the autodiff Gram determinant: at the lobe
/// axis f32 rounds `cosθ` to exactly `1.0`, so `sinθ = √(1−cos²θ) = 0` and the
/// autodiff Jacobian forms `0·∞ = NaN` there — an input-nudge can't escape the
/// rounding for small `α`. The analytic form is finite everywhere (`boundary`
/// tests) and is validated against an independent re-derivation of the closed
/// form on the interior by `ggx_pdf_matches_closed_form`.
#[inline(always)]
pub fn ggx_direction_with_pdf<S: Simd>(
    r: Sample2D,
    alpha: f32,
) -> (Vec3<S>, ScalarPDF<SolidAngle>) {
    let v = ggx_direction::<S>(r, alpha);
    let cos_theta = v.z().max(0.0); // lobe is around +z
    let a2 = alpha * alpha;
    let denom = (a2 - 1.0) * cos_theta * cos_theta + 1.0;
    let d = a2 / (PI * denom * denom);
    let p = d * cos_theta;
    (v, PDF::new(Vector::<f32>::splat(p)))
}

#[deprecated(note = "Ad-hoc normalize-based lobe with no closed-form pdf and not \
            AD-compatible. Redirects to `power_cosine_direction` with n = \
            weight²; use that (or `_pdf` for the automatic solid-angle pdf). \
            NOTE: this CHANGES the produced distribution to a true cosⁿ lobe — \
            exact only at weight = 1 (n = 1, the cosine lobe).")]
#[inline(always)]
pub fn weighted_cosine_direction<S: Simd>(r: Sample2D, weight: f32) -> Vec3<S> {
    power_cosine_direction::<S>(r, weight * weight)
}

#[inline(always)]
pub fn random_to_sphere<S: Simd>(r: Sample2D, radius: f32, distance_squared: f32) -> Vec3<S> {
    let k = radius * radius / distance_squared;
    vec3_of(to_sphere_core::<f32>(open_unit(r.x), open_unit(r.y), k).map(SampleField::value))
}

/// As [`random_to_sphere`], but also returns the (uniform) solid-angle pdf over
/// the subtended spherical cap, computed automatically from the warp Jacobian.
#[inline(always)]
pub fn random_to_sphere_with_pdf<S: Simd>(
    r: Sample2D,
    radius: f32,
    distance_squared: f32,
) -> (Vec3<S>, ScalarPDF<SolidAngle>) {
    let k = radius * radius / distance_squared;
    let (v, p) = warp_with_pdf::<S>(|a, b| to_sphere_core(a, b, k), r);
    (v, PDF::new(Vector::<f32>::splat(p)))
}

/// Closed-form solid-angle pdf of [`random_to_sphere`]: the uniform density over
/// the subtended spherical cap of half-angle `θ_max`, where `cos θ_max = √(1−k)`
/// and `k = radius²/distance²`. Derivation: the cap solid angle is
/// `Ω = 2π(1−cos θ_max)`, and the sampler is uniform over it, so
/// `p_ω = 1/Ω = 1/(2π(1−√(1−k)))`.
///
/// This is the analytic equivalent of the value returned by
/// [`random_to_sphere_with_pdf`]. Unlike the power-cosine lobe, the cap warp's
/// Gram determinant is a nonzero constant everywhere on the cap, so this matches
/// the autodiff result to floating-point precision (no pole singularity). Prefer
/// it when you only need the density for a direction already known to lie inside
/// the cap (e.g. MIS weighting), avoiding the dual-number evaluation.
#[inline(always)]
pub fn to_sphere_pdf(radius: f32, distance_squared: f32) -> ScalarPDF<SolidAngle> {
    let k = radius * radius / distance_squared;
    let cos_theta_max = (1.0 - k).sqrt();
    PDF::new(Vector::<f32>::splat(
        1.0 / (2.0 * PI * (1.0 - cos_theta_max)),
    ))
}

/// As [`random_to_sphere_with_pdf`], but returns the analytic closed-form pdf
/// (see [`to_sphere_pdf`]) instead of the autodiff Gram-determinant value. The
/// direction is generated by the same warp, so the pair is consistent.
#[inline(always)]
pub fn random_to_sphere_with_pdf_analytic<S: Simd>(
    r: Sample2D,
    radius: f32,
    distance_squared: f32,
) -> (Vec3<S>, ScalarPDF<SolidAngle>) {
    (
        random_to_sphere::<S>(r, radius, distance_squared),
        to_sphere_pdf(radius, distance_squared),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // type TestS = thermite::backend::scalar::Scalar;
    type TestS = thermite::backend::x86_v3::X86V3;
    type V3 = Vec3<TestS>;

    fn arb_sample2d() -> impl Strategy<Value = Sample2D> {
        (0.0f32..1.0, 0.0f32..1.0).prop_map(|(x, y)| Sample2D::new(x, y))
    }

    fn arb_sample2d_interior() -> impl Strategy<Value = Sample2D> {
        (0.0f32..1.0, 0.0f32..1.0)
            .prop_map(|(x, y)| Sample2D::new(0.001 + x * 0.998, 0.001 + y * 0.998))
    }

    fn arb_sample3d() -> impl Strategy<Value = Sample3D> {
        (0.0f32..1.0, 0.0f32..1.0, 0.0f32..1.0).prop_map(|(x, y, z)| Sample3D::new(x, y, z))
    }

    /// Interior-restricted `Sample3D`, kept off the spherical-coordinate
    /// singularities (`sinθ→0`, `c^(−2/3)→∞`) where the 3×3 autodiff determinant
    /// loses f32 precision. Built by remapping the unit interval rather than
    /// sampling a sub-unit range directly, which trips a panic in proptest's
    /// bounded-float sampler (`float_samplers.rs`).
    fn arb_sample3d_interior() -> impl Strategy<Value = Sample3D> {
        (0.0f32..1.0, 0.0f32..1.0, 0.0f32..1.0)
            .prop_map(|(x, y, z)| Sample3D::new(0.02 + 0.96 * x, 0.05 + 0.90 * y, 0.05 + 0.93 * z))
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2000000))]

        #[test]
        fn sphere_pdf_is_uniform_solid_angle(s in arb_sample2d()) {
            let (v, p): (V3, ScalarPDF<SolidAngle>) = random_on_unit_sphere_with_pdf(s);
            // pdf is uniform over the sphere: 1/(4π)
            prop_assert!((p.raw().extract::<0>() - 1.0 / (4.0 * PI)).abs() < 1e-4, "p={}", p.raw().extract::<0>());
            // sample agrees with the value-only path
            let v2: V3 = random_on_unit_sphere(s);
            prop_assert!((v - v2).norm() < 1e-5);
        }

        #[test]
        fn in_unit_sphere_norm_le_1(s in arb_sample3d()) {
            let v: V3 = random_in_unit_sphere(s);
            let n = v.norm();
            prop_assert!(n <= 1.0 + 1e-6, "||v||={} > 1", n);
        }

        #[test]
        fn on_unit_sphere_norm_approx_1(s in arb_sample2d()) {
            let v: V3 = random_on_unit_sphere(s);
            let n = v.norm();
            prop_assert!((n - 1.0).abs() < 1e-4, "||v||={}", n);
        }

        #[test]
        fn in_unit_disk_constraints(s in arb_sample2d()) {
            let v: V3 = random_in_unit_disk(s);
            let r2 = v.x() * v.x() + v.y() * v.y();
            prop_assert!(r2 <= 1.0 + 1e-6, "x^2+y^2={} > 1", r2);
            prop_assert!(v.z().abs() < 1e-6, "z={} should be 0", v.z());
        }

        #[test]
        fn cosine_direction_upper_hemisphere(s in arb_sample2d()) {
            let v: V3 = random_cosine_direction(s);
            prop_assert!(v.z() >= 0.0, "z={} < 0", v.z());
        }

        #[test]
        fn cosine_direction_approx_unit(s in arb_sample2d()) {
            let v: V3 = random_cosine_direction(s);
            let n = v.norm();
            prop_assert!((n - 1.0).abs() < 1e-3, "||v||={}", n);
        }

        #[test]
        #[allow(deprecated)]
        fn weighted_cosine_direction_unit(s in arb_sample2d(), w in 0.1f32..2.0) {
            let v: V3 = weighted_cosine_direction(s, w);
            let n = v.norm();
            prop_assert!((n - 1.0).abs() < 1e-3, "||v||={}", n);
        }

        #[test]
        fn random_to_sphere_z_ge_threshold(s in arb_sample2d()) {
            let v: V3 = random_to_sphere(s, 1.0, 4.0);
            let n = v.norm();
            prop_assert!((n - 1.0).abs() < 1e-3, "||v||={}", n);
            let cos_theta_max = (1.0 - 1.0 / 4.0f32).sqrt();
            prop_assert!(v.z() >= cos_theta_max - 1e-3, "z={} < threshold={}", v.z(), cos_theta_max);
        }

        // ---- auto-pdf (dual-number) warps --------------------------------
        // Each asserts the pdf computed from the warp Jacobian matches the
        // known closed form, and that the sample equals the value-path warp.


        #[test]
        fn disk_pdf_is_uniform_area(s in arb_sample2d()) {
            let (v, p): (V3, ScalarPDF<Area>) = random_in_unit_disk_with_pdf(s);
            // pdf is uniform over the unit disk: 1/π
            prop_assert!((p.raw().extract::<0>() - 1.0 / PI).abs() < 1e-4, "p={}", p.raw().extract::<0>());
            let v2: V3 = random_in_unit_disk(s);
            prop_assert!((v - v2).norm() < 1e-5);
        }

        #[test]
        fn cosine_pdf_matches_cos_over_pi(s in arb_sample2d()) {
            let (v, p): (V3, ScalarPDF<SolidAngle>) = random_cosine_direction_with_pdf(s);
            // solid-angle pdf of cosine-weighted sampling is cosθ/π = z/π
            let expected = v.z() / PI;
            prop_assert!((p.raw().extract::<0>() - expected).abs() < 1e-3, "p={}, expected={}", p.raw().extract::<0>(), expected);
            // converting to projected solid angle gives the uniform 1/π
            let p_psa = p.convert(DirectionalGeom { cos_theta: v.z() });
            prop_assert!((p_psa.raw().extract::<0>() - 1.0 / PI).abs() < 1e-3, "p_psa={}", p_psa.raw().extract::<0>());
            let v2: V3 = random_cosine_direction(s);
            prop_assert!((v - v2).norm() < 1e-5);
        }

        // restrict away from the coordinate poles/origin: the spherical-coordinate
        // Jacobian entries blow up there (sinθ→0, c^(-2/3)→∞), so the 3×3 det is
        // formed from huge near-cancelling terms and loses f32 precision — the
        // analytic pdf stays a constant 3/(4π) regardless.
        #[test]
        fn volume_pdf_is_uniform(s in arb_sample3d_interior()) {
            let (v, p): (V3, ScalarPDF<Volume>) = random_in_unit_sphere_with_pdf(s);
            // uniform over the unit ball: 1 / (4/3 π) = 3/(4π)
            prop_assert!((p.raw().extract::<0>() - 3.0 / (4.0 * PI)).abs() < 2e-3, "p={}", p.raw().extract::<0>());
            let v2: V3 = random_in_unit_sphere(s);
            prop_assert!((v - v2).norm() < 1e-5);
        }

        #[test]
        fn to_sphere_pdf_is_uniform_cap(s in arb_sample2d()) {
            let (radius, dist_sq) = (1.0f32, 4.0f32);
            let (v, p): (V3, ScalarPDF<SolidAngle>) = random_to_sphere_with_pdf(s, radius, dist_sq);
            // uniform over the subtended cap: 1 / (2π(1 - cosθ_max))
            let cos_theta_max = (1.0 - radius * radius / dist_sq).sqrt();
            let expected = 1.0 / (2.0 * PI * (1.0 - cos_theta_max));
            prop_assert!((p.raw().extract::<0>() - expected).abs() < 1e-3, "p={}, expected={}", p.raw().extract::<0>(), expected);
            let v2: V3 = random_to_sphere(s, radius, dist_sq);
            prop_assert!((v - v2).norm() < 1e-5);
        }

        // The closed-form `to_sphere_pdf` must reproduce the autodiff Gram-det
        // value across a range of cap sizes (the cap warp has no pole, so they
        // agree to f32 precision), and the analytic sampling variant must return
        // the same direction as the autodiff variant.
        #[test]
        fn to_sphere_pdf_analytic_matches_autodiff(
            s in arb_sample2d_interior(), radius in 0.1f32..3.0, dist_sq in 10.0f32..40.0,
        ) {
            let (v_ad, p_ad): (V3, ScalarPDF<SolidAngle>) =
                random_to_sphere_with_pdf(s, radius, dist_sq);
            let (v_an, p_an): (V3, ScalarPDF<SolidAngle>) =
                random_to_sphere_with_pdf_analytic(s, radius, dist_sq);
            prop_assert!((v_ad - v_an).norm() < 1e-5, "direction mismatch");
            let p_direct = to_sphere_pdf(radius, dist_sq);
            prop_assert!((p_an.raw().extract::<0>() - p_direct.raw().extract::<0>()).abs() < 1e-6);
            // relative tolerance: the density spans a wide range over the cap sizes
            prop_assert!(
                (p_an.raw().extract::<0>() - p_ad.raw().extract::<0>()).abs() <= 1e-4 + 1e-3 * p_ad.raw().extract::<0>(),
                "analytic={}, autodiff={}", p_an.raw().extract::<0>(), p_ad.raw().extract::<0>()
            );
        }

        // ---- concentration-controllable lobe samplers --------------------

        #[test]
        fn power_cosine_pdf_matches_closed_form(
            s in arb_sample2d(), n in 0.0f32..64.0,
        ) {
            let (v, p): (V3, ScalarPDF<SolidAngle>) = power_cosine_direction_with_pdf(s, n);
            // pdf = (n+1)/(2π) · cosⁿθ  (cosθ = z, the lobe is around +z)
            let z = v.z();
            let expected = (n + 1.0) / (2.0 * PI) * z.powf(n);
            // relative tolerance: zⁿ varies over a wide dynamic range
            prop_assert!(
                (p.raw().extract::<0>() - expected).abs() <= 1e-3 + 1e-2 * expected,
                "p={}, expected={}, n={}", p.raw().extract::<0>(), expected, n
            );
            prop_assert!(z >= 0.0, "lobe should be in the upper hemisphere, z={}", z);
            // unit-length by construction — no .normalized() anywhere
            prop_assert!((v.norm() - 1.0).abs() < 1e-5, "||v||={}", v.norm());
            let v2: V3 = power_cosine_direction(s, n);
            prop_assert!((v - v2).norm() < 1e-5);
        }

        #[test]
        fn power_cosine_n1_reduces_to_cosine_density(s in arb_sample2d()) {
            // n=1 is the cosine lobe: pdf = cosθ/π at the produced direction.
            // (The per-sample bijection differs from random_cosine_direction —
            // the two uniform dims play swapped roles — so this checks the
            // density form, not sample equality.)
            let (v, p): (V3, ScalarPDF<SolidAngle>) = power_cosine_direction_with_pdf(s, 1.0);
            prop_assert!((p.raw().extract::<0>() - v.z() / PI).abs() < 1e-4, "p={}, z/π={}", p.raw().extract::<0>(), v.z() / PI);
        }

        #[test]
        fn ggx_pdf_matches_closed_form(
            s in arb_sample2d(), alpha in 0.05f32..1.0,
        ) {
            let (v, p): (V3, ScalarPDF<SolidAngle>) = ggx_direction_with_pdf(s, alpha);
            // pdf = D(θ)·cosθ, D = α²/(π((α²-1)cos²θ+1)²)
            let cos_t = v.z();
            let a2 = alpha * alpha;
            let denom = (a2 - 1.0) * cos_t * cos_t + 1.0;
            let d = a2 / (PI * denom * denom);
            let expected = d * cos_t;
            prop_assert!(
                (p.raw().extract::<0>() - expected).abs() <= 1e-3 + 1e-2 * expected,
                "p={}, expected={}, alpha={}", p.raw().extract::<0>(), expected, alpha
            );
            prop_assert!((v.norm() - 1.0).abs() < 1e-5, "||v||={}", v.norm());
            let v2: V3 = ggx_direction(s, alpha);
            prop_assert!((v - v2).norm() < 1e-5);
        }

        #[test]
        fn ggx_alpha1_reduces_to_cosine_density(s in arb_sample2d()) {
            // at α=1, D=1/π so pdf = cosθ/π at the produced direction.
            let (v, p): (V3, ScalarPDF<SolidAngle>) = ggx_direction_with_pdf(s, 1.0);
            prop_assert!((p.raw().extract::<0>() - v.z() / PI).abs() < 1e-4, "p={}, z/π={}", p.raw().extract::<0>(), v.z() / PI);
        }
    }

    // ======================================================================
    // Deterministic boundary-sample tests.
    //
    // `RandomSampler` yields the half-open interval `[0, 1)`, so `x == 0.0` and
    // `y == 0.0` are *real* RNG outputs (they bit a `sharp_light` render via the
    // sphere pole). A `0.0..1.0` proptest range is half-open and effectively
    // never lands on those exact coordinates, so it does not exercise them.
    //
    // Every autodiff `*_with_pdf` warp computes the pdf as `1/√det(JᵀJ)`. Several
    // cores contain a `√·` (or `acos`, `xᵖ` with `p<1`) whose derivative is `+∞`
    // when its argument is `0` — at a sphere/disk pole or a lobe axis. There the
    // Jacobian has one `∞` column and one zero-length column, so the Gram
    // determinant forms `0·∞ = NaN`. These tests enumerate every {0, ½, 1}²
    // corner/edge sample and assert the pdf is finite and positive, so any warp
    // that degenerates at a boundary is caught deterministically rather than by
    // a lucky proptest draw.
    // ======================================================================
    mod boundary {
        use super::*;

        /// The sampler domain is the half-open `[0, 1)` (`Sample2D::new`
        /// debug-asserts `0 ≤ · < 1`), so the reachable extremes are exactly
        /// `0.0` (a real RNG output — this is the coordinate that NaN'd the warps)
        /// and the largest float strictly below `1.0`. `0.5` is an interior
        /// control point. `1.0` itself is out of contract and never tested.
        const ALMOST_ONE: f32 = f32::from_bits(0x3f7f_ffff); // 1.0 − 1 ulp ≈ 0.99999994
        const EDGE: [f32; 3] = [0.0, 0.5, ALMOST_ONE];

        fn edge_2d() -> Vec<Sample2D> {
            let mut out = Vec::new();
            for &x in &EDGE {
                for &y in &EDGE {
                    out.push(Sample2D::new(x, y));
                }
            }
            out
        }

        fn edge_3d() -> Vec<Sample3D> {
            let mut out = Vec::new();
            for &x in &EDGE {
                for &y in &EDGE {
                    for &z in &EDGE {
                        out.push(Sample3D::new(x, y, z));
                    }
                }
            }
            out
        }

        /// Assert a pdf is a usable density (finite, strictly positive) at a
        /// boundary sample, naming the offending coordinate on failure.
        #[track_caller]
        fn assert_usable_pdf(name: &str, s: impl core::fmt::Debug, p: f32) {
            assert!(
                p.is_finite() && p > 0.0,
                "{name}: pdf degenerated to {p} at boundary sample {s:?} \
                 (expected a finite, positive density)"
            );
        }

        #[test]
        fn sphere_pdf_finite_at_boundaries() {
            for s in edge_2d() {
                let (_v, p): (V3, ScalarPDF<SolidAngle>) = random_on_unit_sphere_with_pdf(s);
                let p = p.raw().extract::<0>();
                assert_usable_pdf("random_on_unit_sphere_with_pdf", s, p);
                // uniform: exact analytic constant everywhere, poles included
                assert!(
                    (p - 1.0 / (4.0 * PI)).abs() < 1e-4,
                    "sphere pdf {p} != 1/(4π) at {s:?}"
                );
            }
        }

        #[test]
        fn disk_pdf_finite_at_boundaries() {
            for s in edge_2d() {
                let (_v, p): (V3, ScalarPDF<Area>) = random_in_unit_disk_with_pdf(s);
                assert_usable_pdf("random_in_unit_disk_with_pdf", s, p.raw().extract::<0>());
            }
        }

        #[test]
        fn cosine_pdf_finite_at_boundaries() {
            for s in edge_2d() {
                let (_v, p): (V3, ScalarPDF<SolidAngle>) = random_cosine_direction_with_pdf(s);
                assert_usable_pdf(
                    "random_cosine_direction_with_pdf",
                    s,
                    p.raw().extract::<0>(),
                );
            }
        }

        #[test]
        fn to_sphere_pdf_finite_at_boundaries() {
            let (radius, dist_sq) = (1.0f32, 4.0f32);
            for s in edge_2d() {
                let (_v, p): (V3, ScalarPDF<SolidAngle>) =
                    random_to_sphere_with_pdf(s, radius, dist_sq);
                assert_usable_pdf("random_to_sphere_with_pdf", s, p.raw().extract::<0>());
            }
        }

        #[test]
        fn power_cosine_pdf_finite_at_boundaries() {
            for n in [0.0f32, 1.0, 27.62905, 64.0] {
                for s in edge_2d() {
                    let (_v, p): (V3, ScalarPDF<SolidAngle>) =
                        power_cosine_direction_with_pdf(s, n);
                    assert_usable_pdf(
                        &format!("power_cosine_direction_with_pdf(n={n})"),
                        s,
                        p.raw().extract::<0>(),
                    );
                }
            }
        }

        #[test]
        fn ggx_pdf_finite_at_boundaries() {
            for alpha in [0.05f32, 0.5, 1.0] {
                for s in edge_2d() {
                    let (_v, p): (V3, ScalarPDF<SolidAngle>) = ggx_direction_with_pdf(s, alpha);
                    assert_usable_pdf(
                        &format!("ggx_direction_with_pdf(alpha={alpha})"),
                        s,
                        p.raw().extract::<0>(),
                    );
                }
            }
        }

        #[test]
        fn volume_pdf_finite_at_boundaries() {
            for s in edge_3d() {
                let (_v, p): (V3, ScalarPDF<Volume>) = random_in_unit_sphere_with_pdf(s);
                let p = p.raw().extract::<0>();
                assert_usable_pdf("random_in_unit_sphere_with_pdf", s, p);
                // uniform over the ball: exact analytic constant everywhere
                assert!(
                    (p - 3.0 / (4.0 * PI)).abs() < 2e-3,
                    "volume pdf {p} != 3/(4π) at {s:?}"
                );
            }
        }

        /// The value path and the pdf path must produce the *same* direction at
        /// every boundary sample (the hardening must nudge both identically).
        #[test]
        fn value_and_pdf_paths_agree_at_boundaries() {
            for s in edge_2d() {
                let (v_pdf, _): (V3, ScalarPDF<SolidAngle>) = random_on_unit_sphere_with_pdf(s);
                let v_val: V3 = random_on_unit_sphere(s);
                assert!(
                    (v_pdf - v_val).norm() < 1e-5,
                    "sphere value/pdf direction mismatch at {s:?}: {v_pdf:?} vs {v_val:?}"
                );
            }
        }
    }
}
