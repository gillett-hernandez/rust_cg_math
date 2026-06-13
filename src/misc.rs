use crate::prelude::*;
use thermite::math::TranscendentalMath;

#[inline(always)]
pub fn power_heuristic(a: f32, b: f32) -> f32 {
    (a * a) / (a * a + b * b)
}

/// Vector form of `power_heuristic`. Replaces the old `power_heuristic_hero`
/// (which was hardcoded to `f32x4`) — now generic across thermite vector widths.
#[inline(always)]
pub fn power_heuristic_v<V: NumericVector>(a: V, b: V) -> V {
    (a * a) / (a * a + b * b)
}

/// MIS power heuristic (β = 2, Veach eq. 9.13) for two sampling strategies whose
/// pdfs are densities w.r.t. the **same** measure `M`. The shared `M` is enforced
/// at compile time, so a solid-angle pdf cannot be weighted against an area pdf:
/// Veach §9.3 requires every strategy's pdf to be expressed against one common
/// measure before combining (convert them with [`PDF::convert`] first). The
/// weight is a dimensionless ratio, so the measure tag drops off the result.
///
/// Generic over the field `T`, so it serves both scalar (`f32`) pdfs and
/// hero-wavelength vector pdfs (per-lane weights), mirroring [`power_heuristic`]
/// / [`power_heuristic_v`], to which it delegates the arithmetic.
///
/// ```
/// use math::prelude::*;
/// let a: PDF<f32, Area> = PDF::new(3.0);
/// let b: PDF<f32, Area> = PDF::new(4.0);
/// let w = power_heuristic_pdf(a, b); // measures match → OK
/// assert!((w - 9.0 / 25.0).abs() < 1e-6);
/// ```
///
/// Mixing measures is rejected by the compiler:
///
/// ```compile_fail
/// use math::prelude::*;
/// let a: PDF<f32, Area> = PDF::new(3.0);
/// let b: PDF<f32, SolidAngle> = PDF::new(4.0);
/// let _w = power_heuristic_pdf(a, b); // ERROR: Area ≠ SolidAngle
/// ```
#[inline(always)]
pub fn power_heuristic_pdf<T: Field, M: Measure>(a: PDF<T, M>, b: PDF<T, M>) -> T {
    let (a, b) = (*a, *b);
    (a * a) / (a * a + b * b)
}

#[inline(always)]
pub fn gaussianf32(x: f32, alpha: f32, mu: f32, sigma1: f32, sigma2: f32) -> f32 {
    let sqrt = (x - mu) / (if x < mu { sigma1 } else { sigma2 });
    alpha * (-(sqrt * sqrt) / 2.0).exp()
}

#[inline(always)]
pub fn gaussian(x: f64, alpha: f64, mu: f64, sigma1: f64, sigma2: f64) -> f64 {
    let sqrt = (x - mu) / (if x < mu { sigma1 } else { sigma2 });
    alpha * (-(sqrt * sqrt) / 2.0).exp()
}

/// Vector form of `gaussianf32`. Generic over any thermite f32 float vector
/// with transcendental support. Replaces the simdfloat_patch-gated
/// `gaussian_f32x4`.
#[inline(always)]
pub fn gaussian_v<V>(x: V, alpha: f32, mu: f32, sigma1: f32, sigma2: f32) -> V
where
    V: FloatVectorWithBits<Element = f32> + TranscendentalMath,
{
    let sigma = x
        .cmp_lt(V::splat(mu))
        .select(V::splat(sigma1), V::splat(sigma2));
    let sqrt = (x - V::splat(mu)) / sigma;
    V::splat(alpha) * (-(sqrt * sqrt) / V::splat(2.0)).exp()
}

#[inline(always)]
pub fn w(x: f32, mul: f32, offset: f32, sigma: f32) -> f32 {
    mul * (-(x - offset).powi(2) / sigma).exp() / (sigma * PI).sqrt()
}

const HCC2: f32 = 1.1910429723971884140794892e-29;
const HKC: f32 = 1.438777085924334052222404423195819240925e-2;

#[inline(always)]
pub fn blackbody(temperature: f32, lambda: f32) -> f32 {
    let lambda = lambda * 1e-9;

    lambda.powi(-5) * HCC2 / ((HKC / (lambda * temperature)).exp() - 1.0)
}

/// Vector form of `blackbody`. Replaces the simdfloat_patch-gated
/// `blackbody_f32x4`; now works across thermite vector widths.
#[inline(always)]
pub fn blackbody_v<V>(temperature: f32, lambda: V) -> V
where
    V: FloatVectorWithBits<Element = f32> + TranscendentalMath,
{
    let lambda = lambda * V::splat(1e-9);
    lambda.powf(V::splat(-5.0)) * V::splat(HCC2)
        / ((V::splat(HKC) / (lambda * V::splat(temperature))).exp() - V::splat(1.0))
}

#[inline(always)]
pub fn max_blackbody_lambda(temp: f32) -> f32 {
    2.8977721e-3 / (temp * 1e-9)
}

//----------------------------------------------------------------------
// theta = azimuthal angle
// phi = inclination, i.e. angle measured from +Z. the elevation angle would be pi/2 - phi

#[inline(always)]
pub fn uv_to_direction<S: thermite::simd::Simd>(uv: (f32, f32)) -> Vec3<S> {
    let theta = (uv.0 - 0.5) * 2.0 * PI;
    let phi = uv.1 * PI;

    let (sin_theta, cos_theta) = theta.sin_cos();
    let (sin_phi, cos_phi) = phi.sin_cos();

    let (x, y, z) = (sin_phi * cos_theta, sin_phi * sin_theta, cos_phi);
    Vec3::new(x, y, z)
}

#[inline(always)]
pub fn direction_to_uv<S: thermite::simd::Simd>(direction: Vec3<S>) -> (f32, f32) {
    let theta = direction.y().atan2(direction.x());
    let phi = direction.z().acos();
    let u = theta / 2.0 / PI + 0.5;
    let v = phi / PI;
    (u, v)
}

//----------------------------------------------------------------------
// Signed distance to an axis-aligned ellipse.
//
// Port of Inigo Quilez's `sdEllipse` (the iterative variant), which refines a
// foot-point on the ellipse with three fixed-point iterations and reports the
// distance to it, negated when the query point is inside. See
// https://iquilezles.org/articles/ellipsedist/ . The sign test compares
// |p|² against |nearest|² rather than a true inside test, which matches the
// reference and is exact away from the (degenerate) evolute cusps.

const FRAC_1_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Scalar signed distance from point `p` to the axis-aligned ellipse with
/// semi-axes `e = (a, b)`. Negative inside, positive outside.
#[inline(always)]
pub fn sd_ellipse(p: (f32, f32), e: (f32, f32)) -> f32 {
    let (pax, pay) = (p.0.abs(), p.1.abs());
    let (eix, eiy) = (1.0 / e.0, 1.0 / e.1);
    let (e2x, e2y) = (e.0 * e.0, e.1 * e.1);
    // ve = ei * (e2.x - e2.y, e2.y - e2.x)
    let (vex, vey) = (eix * (e2x - e2y), eiy * (e2y - e2x));

    let (mut tx, mut ty) = (FRAC_1_SQRT_2, FRAC_1_SQRT_2);
    for _ in 0..3 {
        // v = ve * t^3
        let (vx, vy) = (vex * tx * tx * tx, vey * ty * ty * ty);
        // u = normalize(pAbs - v) * length(t*e - v)
        let (dx, dy) = (pax - vx, pay - vy);
        let inv_d = 1.0 / (dx * dx + dy * dy).sqrt();
        let (lx, ly) = (tx * e.0 - vx, ty * e.1 - vy);
        let len_l = (lx * lx + ly * ly).sqrt();
        let (ux, uy) = (dx * inv_d * len_l, dy * inv_d * len_l);
        // w = ei * (v + u); t = normalize(clamp(w, 0, 1))
        let (wx, wy) = (eix * (vx + ux), eiy * (vy + uy));
        let (cx, cy) = (wx.clamp(0.0, 1.0), wy.clamp(0.0, 1.0));
        let inv_c = 1.0 / (cx * cx + cy * cy).sqrt();
        tx = cx * inv_c;
        ty = cy * inv_c;
    }

    let (nax, nay) = (tx * e.0, ty * e.1);
    let dist = ((pax - nax) * (pax - nax) + (pay - nay) * (pay - nay)).sqrt();
    if pax * pax + pay * pay < nax * nax + nay * nay {
        -dist
    } else {
        dist
    }
}

/// Vector form of [`sd_ellipse`]: evaluates the signed distance for one query
/// point per SIMD lane (`px`/`py` hold the per-lane x/y coordinates) against a
/// single ellipse with semi-axes `e = (a, b)`. The ellipse-derived constants are
/// scalar and splatted once; the per-point work (the three refinement
/// iterations and the final sign select) runs across all lanes at once.
#[inline(always)]
pub fn sd_ellipse_v<V>(px: V, py: V, e: (f32, f32)) -> V
where
    V: FloatVectorWithBits<Element = f32>,
{
    let pax = px.abs();
    let pay = py.abs();
    let (eix, eiy) = (1.0 / e.0, 1.0 / e.1);
    let (e2x, e2y) = (e.0 * e.0, e.1 * e.1);
    let vex = V::splat(eix * (e2x - e2y));
    let vey = V::splat(eiy * (e2y - e2x));
    let (ex, ey) = (V::splat(e.0), V::splat(e.1));
    let (eix_v, eiy_v) = (V::splat(eix), V::splat(eiy));
    let zero = V::splat(0.0);
    let one = V::splat(1.0);

    let mut tx = V::splat(FRAC_1_SQRT_2);
    let mut ty = V::splat(FRAC_1_SQRT_2);
    for _ in 0..3 {
        let vx = vex * tx * tx * tx;
        let vy = vey * ty * ty * ty;
        let dx = pax - vx;
        let dy = pay - vy;
        let inv_d = (dx * dx + dy * dy).sqrt().rcp();
        let lx = tx * ex - vx;
        let ly = ty * ey - vy;
        let len_l = (lx * lx + ly * ly).sqrt();
        let ux = dx * inv_d * len_l;
        let uy = dy * inv_d * len_l;
        let wx = eix_v * (vx + ux);
        let wy = eiy_v * (vy + uy);
        let cx = wx.clamp(zero, one);
        let cy = wy.clamp(zero, one);
        let inv_c = (cx * cx + cy * cy).sqrt().rcp();
        tx = cx * inv_c;
        ty = cy * inv_c;
    }

    let nax = tx * ex;
    let nay = ty * ey;
    let ddx = pax - nax;
    let ddy = pay - nay;
    let dist = (ddx * ddx + ddy * ddy).sqrt();
    // negate where the point is "inside" (|p|² < |nearest|²), matching the scalar form.
    let inside = (pax * pax + pay * pay).cmp_lt(nax * nax + nay * nay);
    inside.select(-dist, dist)
}




#[cfg(test)]
mod test {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn power_heuristic_in_unit_range(a in 0.0f32..100.0, b in 0.0f32..100.0) {
            prop_assume!(a + b > 1e-6);
            let h = power_heuristic(a, b);
            prop_assert!(h >= 0.0 && h <= 1.0, "power_heuristic({}, {}) = {}", a, b, h);
        }

        #[test]
        fn power_heuristic_complement(a in 0.01f32..100.0, b in 0.01f32..100.0) {
            let h1 = power_heuristic(a, b);
            let h2 = power_heuristic(b, a);
            let sum = h1 + h2;
            prop_assert!((sum - 1.0).abs() < 1e-4, "h(a,b)+h(b,a)={}", sum);
        }

        #[test]
        fn power_heuristic_pdf_matches_raw(a in 0.01f32..100.0, b in 0.01f32..100.0) {
            // the measure-tagged form must compute exactly the same weight as the
            // raw primitive once the measure tag is applied.
            let pa: PDF<f32, Area> = PDF::new(a);
            let pb: PDF<f32, Area> = PDF::new(b);
            let w = power_heuristic_pdf(pa, pb);
            prop_assert_eq!(w, power_heuristic(a, b));
        }

        #[test]
        fn power_heuristic_pdf_complement(a in 0.01f32..100.0, b in 0.01f32..100.0) {
            // two pdfs against the SAME measure: weights still sum to 1.
            let pa: PDF<f32, Area> = PDF::new(a);
            let pb: PDF<f32, Area> = PDF::new(b);
            let sum = power_heuristic_pdf(pa, pb) + power_heuristic_pdf(pb, pa);
            prop_assert!((sum - 1.0).abs() < 1e-4, "w(a,b)+w(b,a)={}", sum);
        }

        #[test]
        fn blackbody_non_negative(temp in 1000.0f32..10000.0, lambda in 300.0f32..900.0) {
            let val = blackbody(temp, lambda);
            prop_assert!(val >= 0.0, "blackbody({}, {}) = {}", temp, lambda, val);
        }

        #[test]
        fn max_blackbody_lambda_decreases_with_temp(t1 in 1000.0f32..5000.0, t2 in 5001.0f32..10000.0) {
            let l1 = max_blackbody_lambda(t1);
            let l2 = max_blackbody_lambda(t2);
            prop_assert!(l1 > l2, "Wien: peak({})={} should be > peak({})={}", t1, l1, t2, l2);
        }

        #[test]
        fn gaussian_non_negative_positive_alpha(x in -100.0f64..100.0) {
            let val = gaussian(x, 1.0, 0.0, 10.0, 10.0);
            prop_assert!(val >= 0.0, "gaussian({})={}", x, val);
        }

        #[test]
        fn power_heuristic_v_matches_scalar(a in 0.01f32..100.0, b in 0.01f32..100.0) {
            type TestR = <thermite::backend::scalar::Scalar as thermite::simd::Simd>::f32x4;
            let r = power_heuristic_v(Vector::<TestR>::splat(a), Vector::<TestR>::splat(b));
            let s = power_heuristic(a, b);
            for lane in r.into_array() {
                prop_assert!((lane - s).abs() < 1e-5, "lane {} vs scalar {}", lane, s);
            }
        }

        #[test]
        fn gaussian_v_matches_scalar(x in -100.0f32..100.0) {
            type TestR = <thermite::backend::scalar::Scalar as thermite::simd::Simd>::f32x4;
            let (alpha, mu, s1, s2) = (1.0f32, 5.0f32, 10.0f32, 20.0f32);
            let r = gaussian_v(Vector::<TestR>::splat(x), alpha, mu, s1, s2);
            let s = gaussianf32(x, alpha, mu, s1, s2);
            for lane in r.into_array() {
                prop_assert!((lane - s).abs() < 1e-4, "lane {} vs scalar {}", lane, s);
            }
        }

        #[test]
        fn blackbody_v_matches_scalar(temp in 1000.0f32..10000.0, lambda in 300.0f32..900.0) {
            type TestR = <thermite::backend::scalar::Scalar as thermite::simd::Simd>::f32x4;
            let r = blackbody_v(temp, Vector::<TestR>::splat(lambda));
            let s = blackbody(temp, lambda);
            let tol = (s.abs() * 1e-3).max(1e-6);
            for lane in r.into_array() {
                prop_assert!((lane - s).abs() < tol, "lane {} vs scalar {}", lane, s);
            }
        }

        #[test]
        fn w_peaks_at_offset(offset in -10.0f32..10.0, x in -10.0f32..10.0, sigma in 0.5f32..5.0) {
            // the asymmetric gaussian-ish `w` is positive and maximal at x == offset.
            let at_peak = w(offset, 1.0, offset, sigma);
            let at_x = w(x, 1.0, offset, sigma);
            prop_assert!(at_x >= 0.0, "w should be non-negative, got {}", at_x);
            prop_assert!(at_peak >= at_x - 1e-6, "peak {} < w(x) {}", at_peak, at_x);
        }

        #[test]
        fn uv_to_direction_unit_length(u in 0.01f32..0.99, v in 0.01f32..0.99) {
            let dir: V3 = uv_to_direction((u, v));
            let n = dir.norm();
            prop_assert!((n - 1.0).abs() < 1e-4, "||dir||={}", n);
        }

        #[test]
        fn uv_direction_roundtrip(u in 0.01f32..0.99, v in 0.01f32..0.99) {
            let dir: V3 = uv_to_direction((u, v));
            let (u2, v2) = direction_to_uv(dir);
            let err_u = (u - u2).abs();
            let err_v = (v - v2).abs();
            // skip poles where atan2 is degenerate
            prop_assume!(v > 0.01 && v < 0.99);
            prop_assert!(err_u < 1e-3, "u roundtrip: {} -> {}", u, u2);
            prop_assert!(err_v < 1e-3, "v roundtrip: {} -> {}", v, v2);
        }
    }

    type TestS = thermite::backend::scalar::Scalar;
    type V3 = Vec3<TestS>;
    type TestR = <thermite::backend::scalar::Scalar as thermite::simd::Simd>::f32x4;

    proptest! {
        // A point on the ellipse boundary has (near) zero signed distance.
        #[test]
        fn sd_ellipse_zero_on_boundary(
            t in 0.0f32..std::f32::consts::TAU,
            a in 0.5f32..5.0,
            b in 0.5f32..5.0,
        ) {
            let (s, c) = t.sin_cos();
            let p = (a * c, b * s);
            let d = sd_ellipse(p, (a, b));
            prop_assert!(d.abs() < 1e-2, "boundary dist {} for t={}, e=({},{})", d, t, a, b);
        }

        // Inside points are negative, outside points are positive (use the unit
        // circle, where the signed distance has a closed form: |p| - 1).
        #[test]
        fn sd_ellipse_sign_on_circle(x in -3.0f32..3.0, y in -3.0f32..3.0) {
            let r = x.hypot(y);
            prop_assume!((r - 1.0).abs() > 1e-2); // skip points right on the boundary
            let d = sd_ellipse((x, y), (1.0, 1.0));
            prop_assert_eq!(d < 0.0, r < 1.0, "sign mismatch: |p|={}, d={}", r, d);
            // for a circle the exact distance is |p| - 1
            prop_assert!((d - (r - 1.0)).abs() < 1e-3, "circle dist {} vs {}", d, r - 1.0);
        }

        // The vectorized form must agree with the scalar form lane-by-lane.
        #[test]
        fn sd_ellipse_v_matches_scalar(
            x in -4.0f32..4.0, y in -4.0f32..4.0,
            a in 0.5f32..5.0, b in 0.5f32..5.0,
        ) {
            let v = sd_ellipse_v::<Vector<TestR>>(
                Vector::<TestR>::splat(x),
                Vector::<TestR>::splat(y),
                (a, b),
            );
            let s = sd_ellipse((x, y), (a, b));
            for lane in v.into_array() {
                prop_assert!((lane - s).abs() < 1e-4, "lane {} vs scalar {}", lane, s);
            }
        }
    }

    #[test]
    fn test_direction_to_uv() {
        let direction: V3 = random_on_unit_sphere(Sample2D::new_random_sample());
        let uv = direction_to_uv(direction);
        assert!(uv.0 >= 0.0 && uv.0 <= 1.0, "u out of range: {}", uv.0);
        assert!(uv.1 >= 0.0 && uv.1 <= 1.0, "v out of range: {}", uv.1);
    }

    #[test]
    fn test_uv_to_direction() {
        let mut center = V3::ZERO;
        let n = 100;
        for _ in 0..n {
            let uv = (debug_random(), debug_random());
            let direction: V3 = uv_to_direction(uv);
            let norm = direction.norm();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "direction not unit length: norm = {}",
                norm
            );
            center = center + direction / n as f32;
        }
        assert!(
            center.norm() < 0.5,
            "center of random directions too far from zero: {:?}",
            center
        );
    }

    #[test]
    fn test_bijectiveness_of_uv_direction() {
        let sub = |a: (f32, f32), b: (f32, f32)| (a.0 - b.0, a.1 - b.1);
        for _ in 0..10000 {
            let uv = (debug_random(), debug_random());
            let direction: V3 = uv_to_direction(uv);
            let uv2 = direction_to_uv(direction);
            let abs_error = sub(uv, uv2);
            let round_trip_error = abs_error.0.hypot(abs_error.1);
            if uv2.1 == 0.0 || uv.1 == 0.0 {
                continue;
            }
            assert!(
                round_trip_error < 0.0001,
                "{:?} {:?}, {:?}, direction = {:?}",
                uv,
                uv2,
                round_trip_error,
                direction
            );

            let direction: V3 = random_on_unit_sphere(Sample2D::new_random_sample());
            let uv = direction_to_uv(direction);
            let direction2: V3 = uv_to_direction(uv);
            let round_trip_error = (direction - direction2).norm();
            assert!(
                round_trip_error < 0.0001,
                "{:?} {:?}, {:?}",
                direction,
                direction2,
                round_trip_error
            );
        }
    }
}
