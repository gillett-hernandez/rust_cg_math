use crate::prelude::*;

pub fn power_heuristic(a: f32, b: f32) -> f32 {
    (a * a) / (a * a + b * b)
}

pub fn power_heuristic_hero(a: f32x4, b: f32x4) -> f32x4 {
    (a * a) / (a * a + b * b)
}

pub fn gaussianf32(x: f32, alpha: f32, mu: f32, sigma1: f32, sigma2: f32) -> f32 {
    let sqrt = (x - mu) / (if x < mu { sigma1 } else { sigma2 });
    alpha * (-(sqrt * sqrt) / 2.0).exp()
}

pub fn gaussian(x: f64, alpha: f64, mu: f64, sigma1: f64, sigma2: f64) -> f64 {
    let sqrt = (x - mu) / (if x < mu { sigma1 } else { sigma2 });
    alpha * (-(sqrt * sqrt) / 2.0).exp()

}

#[cfg(feature="simdfloat_patch")]
pub fn gaussian_f32x4(x: f32x4, alpha: f32, mu: f32, sigma1: f32, sigma2: f32) -> f32x4 {
    use std::simd::Select;

    let sqrt = (x - f32x4::splat(mu))
        / x.simd_lt(f32x4::splat(mu))
            .select(f32x4::splat(sigma1), f32x4::splat(sigma2));

    f32x4::splat(alpha) * (-(sqrt * sqrt) / f32x4::splat(2.0)).exp()

}

pub fn w(x: f32, mul: f32, offset: f32, sigma: f32) -> f32 {
    mul * (-(x - offset).powi(2) / sigma).exp() / (sigma * PI).sqrt()
}

const HCC2: f32 = 1.1910429723971884140794892e-29;
const HKC: f32 = 1.438777085924334052222404423195819240925e-2;

pub fn blackbody(temperature: f32, lambda: f32) -> f32 {
    let lambda = lambda * 1e-9;

    lambda.powi(-5) * HCC2 / ((HKC / (lambda * temperature)).exp() - 1.0)
}

#[cfg(feature="simdfloat_patch")]
pub fn blackbody_f32x4(temperature: f32, lambda: f32x4) -> f32x4 {
    let lambda = lambda * f32x4::splat(1e-9);

    lambda.powf(f32x4::splat(-5.0)) * f32x4::splat(HCC2)
        / ((f32x4::splat(HKC) / (lambda * f32x4::splat(temperature))).exp() - f32x4::splat(1.0))
}

pub fn max_blackbody_lambda(temp: f32) -> f32 {
    2.8977721e-3 / (temp * 1e-9)
}

//----------------------------------------------------------------------
// theta = azimuthal angle
// phi = inclination, i.e. angle measured from +Z. the elevation angle would be pi/2 - phi

pub fn uv_to_direction(uv: (f32, f32)) -> Vec3 {
    let theta = (uv.0 - 0.5) * 2.0 * PI;
    let phi = uv.1 * PI;

    let (sin_theta, cos_theta) = theta.sin_cos();
    let (sin_phi, cos_phi) = phi.sin_cos();

    let (x, y, z) = (sin_phi * cos_theta, sin_phi * sin_theta, cos_phi);
    Vec3::new(x, y, z)
}

pub fn direction_to_uv(direction: Vec3) -> (f32, f32) {
    let theta = direction.y().atan2(direction.x());
    let phi = direction.z().acos();
    let u = theta / 2.0 / PI + 0.5;
    let v = phi / PI;
    (u, v)
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
        fn uv_to_direction_unit_length(u in 0.01f32..0.99, v in 0.01f32..0.99) {
            let dir = uv_to_direction((u, v));
            let n = dir.norm();
            prop_assert!((n - 1.0).abs() < 1e-4, "||dir||={}", n);
        }

        #[test]
        fn uv_direction_roundtrip(u in 0.01f32..0.99, v in 0.01f32..0.99) {
            let dir = uv_to_direction((u, v));
            let (u2, v2) = direction_to_uv(dir);
            let err_u = (u - u2).abs();
            let err_v = (v - v2).abs();
            // skip poles where atan2 is degenerate
            prop_assume!(v > 0.01 && v < 0.99);
            prop_assert!(err_u < 1e-3, "u roundtrip: {} -> {}", u, u2);
            prop_assert!(err_v < 1e-3, "v roundtrip: {} -> {}", v, v2);
        }
    }

    #[test]
    fn test_direction_to_uv() {
        let direction = random_on_unit_sphere(Sample2D::new_random_sample());
        let uv = direction_to_uv(direction);
        println!("{:?} {:?}", direction, uv);
    }

    #[test]
    fn test_uv_to_direction() {
        let mut center = Vec3::ZERO;
        let n = 100;
        for _ in 0..n {
            let uv = (debug_random(), debug_random());
            let direction = uv_to_direction(uv);
            println!("{:?} {:?}", direction, uv);
            center = center + direction / n as f32;
        }
        println!("{:?}", center);
    }

    #[test]
    fn test_bijectiveness_of_uv_direction() {
        let sub = |a: (f32, f32), b: (f32, f32)| (a.0 - b.0, a.1 - b.1);
        for _ in 0..1000000 {
            let uv = (debug_random(), debug_random());
            let direction = uv_to_direction(uv);
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

            let direction = random_on_unit_sphere(Sample2D::new_random_sample());
            let uv = direction_to_uv(direction);
            let direction2 = uv_to_direction(uv);
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
