use crate::prelude::*;
use thermite::register::LinAlg3Register;
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

#[inline(always)]
fn vec3_of<S: Simd>(c: [f32; 3]) -> Vec3<S> {
    Vec3::new(c[0], c[1], c[2])
}

/// Run a 2-input warp core on dual inputs and split into (sample, pdf-value),
/// where the pdf value is the reciprocal Gram determinant of the warp Jacobian.
#[inline(always)]
fn warp_with_pdf<S: Simd>(
    core: impl Fn(Dual<2>, Dual<2>) -> [Dual<2>; 3],
    r: Sample2D,
) -> (Vec3<S>, f32) {
    let out = core(Dual::variable(r.x, 0), Dual::variable(r.y, 1));
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
        Dual::variable(r.x, 0),
        Dual::variable(r.y, 1),
        Dual::variable(r.z, 2),
    );
    let v = vec3_of([out[0].value(), out[1].value(), out[2].value()]);
    (v, reciprocal_det_3(&out))
}

/// Uniformly distributed wrt the volume measure on the unit ball.
#[inline(always)]
pub fn random_in_unit_sphere<S: Simd>(r: Sample3D) -> Vec3<S> {
    vec3_of(in_unit_sphere_core::<f32>(r.x, r.y, r.z).map(SampleField::value))
}

/// As [`random_in_unit_sphere`], but also returns the (uniform) volume pdf
/// (`3/(4π)`) computed automatically from the warp Jacobian.
#[inline(always)]
pub fn random_in_unit_sphere_pdf<S: Simd>(r: Sample3D) -> (Vec3<S>, PDF<f32, Volume>) {
    let (v, p) = warp_with_pdf_3::<S>(in_unit_sphere_core, r);
    (v, PDF::new(p))
}

/// Uniformly distributed wrt the surface area / solid angle measure.
#[inline(always)]
pub fn random_on_unit_sphere<S: Simd>(r: Sample2D) -> Vec3<S> {
    vec3_of(on_unit_sphere_core::<f32>(r.x, r.y).map(SampleField::value))
}

/// As [`random_on_unit_sphere`], but also returns the (uniform) solid-angle pdf
/// computed automatically from the warp Jacobian.
#[inline(always)]
pub fn random_on_unit_sphere_pdf<S: Simd>(r: Sample2D) -> (Vec3<S>, PDF<f32, SolidAngle>) {
    let (v, p) = warp_with_pdf::<S>(on_unit_sphere_core, r);
    (v, PDF::new(p))
}

/// Uniformly distributed wrt the area measure on the unit disk.
#[inline(always)]
pub fn random_in_unit_disk<S: Simd>(r: Sample2D) -> Vec3<S> {
    vec3_of(in_unit_disk_core::<f32>(r.x, r.y).map(SampleField::value))
}

/// As [`random_in_unit_disk`], but also returns the (uniform) area pdf computed
/// automatically from the warp Jacobian.
#[inline(always)]
pub fn random_in_unit_disk_pdf<S: Simd>(r: Sample2D) -> (Vec3<S>, PDF<f32, Area>) {
    let (v, p) = warp_with_pdf::<S>(in_unit_disk_core, r);
    (v, PDF::new(p))
}

/// Cosine-weighted hemisphere direction. Uniform wrt projected solid angle.
#[inline(always)]
pub fn random_cosine_direction<S: Simd>(r: Sample2D) -> Vec3<S> {
    vec3_of(cosine_direction_core::<f32>(r.x, r.y).map(SampleField::value))
}

/// As [`random_cosine_direction`], but also returns the solid-angle pdf
/// (`cosθ/π`) computed automatically from the warp Jacobian. Convert to a
/// projected-solid-angle pdf (`1/π`) with `pdf.convert(DirectionalGeom { .. })`.
#[inline(always)]
pub fn random_cosine_direction_pdf<S: Simd>(r: Sample2D) -> (Vec3<S>, PDF<f32, SolidAngle>) {
    let (v, p) = warp_with_pdf::<S>(cosine_direction_core, r);
    (v, PDF::new(p))
}

#[inline(always)]
pub fn weighted_cosine_direction<S: Simd>(r: Sample2D, weight: f32) -> Vec3<S>
where
    S::f32x4: LinAlg3Register,
{
    let Sample2D { x: u, y: v } = r;
    let z: f32 = weight * (1.0 - v).sqrt();
    let phi: f32 = 2.0 * PI * u;
    let (mut y, mut x) = phi.sin_cos();
    x *= v.sqrt();
    y *= v.sqrt();
    Vec3::new(x, y, z).normalized()
}

#[inline(always)]
pub fn random_to_sphere<S: Simd>(r: Sample2D, radius: f32, distance_squared: f32) -> Vec3<S> {
    let k = radius * radius / distance_squared;
    vec3_of(to_sphere_core::<f32>(r.x, r.y, k).map(SampleField::value))
}

/// As [`random_to_sphere`], but also returns the (uniform) solid-angle pdf over
/// the subtended spherical cap, computed automatically from the warp Jacobian.
#[inline(always)]
pub fn random_to_sphere_pdf<S: Simd>(
    r: Sample2D,
    radius: f32,
    distance_squared: f32,
) -> (Vec3<S>, PDF<f32, SolidAngle>) {
    let k = radius * radius / distance_squared;
    let (v, p) = warp_with_pdf::<S>(|a, b| to_sphere_core(a, b, k), r);
    (v, PDF::new(p))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    type TestS = thermite::backend::scalar::Scalar;
    type V3 = Vec3<TestS>;

    fn arb_sample2d() -> impl Strategy<Value = Sample2D> {
        (0.0f32..0.9999, 0.0f32..0.9999).prop_map(|(x, y)| Sample2D::new(x, y))
    }

    fn arb_sample3d() -> impl Strategy<Value = Sample3D> {
        (0.0f32..0.9999, 0.0f32..0.9999, 0.0f32..0.9999)
            .prop_map(|(x, y, z)| Sample3D::new(x, y, z))
    }

    proptest! {
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
        fn sphere_pdf_is_uniform_solid_angle(s in arb_sample2d()) {
            let (v, p): (V3, PDF<f32, SolidAngle>) = random_on_unit_sphere_pdf(s);
            // pdf is uniform over the sphere: 1/(4π)
            prop_assert!((*p - 1.0 / (4.0 * PI)).abs() < 1e-4, "p={}", *p);
            // sample agrees with the value-only path
            let v2: V3 = random_on_unit_sphere(s);
            prop_assert!((v - v2).norm() < 1e-5);
        }

        #[test]
        fn disk_pdf_is_uniform_area(s in arb_sample2d()) {
            let (v, p): (V3, PDF<f32, Area>) = random_in_unit_disk_pdf(s);
            // pdf is uniform over the unit disk: 1/π
            prop_assert!((*p - 1.0 / PI).abs() < 1e-4, "p={}", *p);
            let v2: V3 = random_in_unit_disk(s);
            prop_assert!((v - v2).norm() < 1e-5);
        }

        #[test]
        fn cosine_pdf_matches_cos_over_pi(s in arb_sample2d()) {
            let (v, p): (V3, PDF<f32, SolidAngle>) = random_cosine_direction_pdf(s);
            // solid-angle pdf of cosine-weighted sampling is cosθ/π = z/π
            let expected = v.z() / PI;
            prop_assert!((*p - expected).abs() < 1e-3, "p={}, expected={}", *p, expected);
            // converting to projected solid angle gives the uniform 1/π
            let p_psa = p.convert(DirectionalGeom { cos_theta: v.z() });
            prop_assert!((*p_psa - 1.0 / PI).abs() < 1e-3, "p_psa={}", *p_psa);
            let v2: V3 = random_cosine_direction(s);
            prop_assert!((v - v2).norm() < 1e-5);
        }

        // restrict away from the coordinate poles/origin: the spherical-coordinate
        // Jacobian entries blow up there (sinθ→0, c^(-2/3)→∞), so the 3×3 det is
        // formed from huge near-cancelling terms and loses f32 precision — the
        // analytic pdf stays a constant 3/(4π) regardless.
        #[test]
        fn volume_pdf_is_uniform(
            x in 0.02f32..0.98, y in 0.05f32..0.95, z in 0.05f32..0.98,
        ) {
            let s = Sample3D::new(x, y, z);
            let (v, p): (V3, PDF<f32, Volume>) = random_in_unit_sphere_pdf(s);
            // uniform over the unit ball: 1 / (4/3 π) = 3/(4π)
            prop_assert!((*p - 3.0 / (4.0 * PI)).abs() < 2e-3, "p={}", *p);
            let v2: V3 = random_in_unit_sphere(s);
            prop_assert!((v - v2).norm() < 1e-5);
        }

        #[test]
        fn to_sphere_pdf_is_uniform_cap(s in arb_sample2d()) {
            let (radius, dist_sq) = (1.0f32, 4.0f32);
            let (v, p): (V3, PDF<f32, SolidAngle>) = random_to_sphere_pdf(s, radius, dist_sq);
            // uniform over the subtended cap: 1 / (2π(1 - cosθ_max))
            let cos_theta_max = (1.0 - radius * radius / dist_sq).sqrt();
            let expected = 1.0 / (2.0 * PI * (1.0 - cos_theta_max));
            prop_assert!((*p - expected).abs() < 1e-3, "p={}, expected={}", *p, expected);
            let v2: V3 = random_to_sphere(s, radius, dist_sq);
            prop_assert!((v - v2).norm() < 1e-5);
        }
    }
}
