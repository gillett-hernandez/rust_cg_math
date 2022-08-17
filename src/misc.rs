use crate::prelude::*;
use packed_simd::f32x4;

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

pub fn gaussian_f32x4(x: f32x4, alpha: f32, mu: f32, sigma1: f32, sigma2: f32) -> f32x4 {
    let sqrt = (x - mu)
        / x.lt(f32x4::splat(mu))
            .select(f32x4::splat(sigma1), f32x4::splat(sigma2));
    alpha * (-(sqrt * sqrt) / 2.0).exp()
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

pub fn blackbody_f32x4(temperature: f32, lambda: f32x4) -> f32x4 {
    let lambda = lambda * 1e-9;

    lambda.powf(f32x4::splat(-5.0)) * HCC2 / ((HKC / (lambda * temperature)).exp() - 1.0)
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
    use crate::sample::Sample2D;

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
            assert!(
                round_trip_error < 0.0001,
                "{:?} {:?}, {:?}",
                uv,
                uv2,
                round_trip_error
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
