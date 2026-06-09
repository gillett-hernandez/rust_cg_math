use crate::prelude::*;
use thermite::simd::Simd;
use thermite::register::LinAlg3Register;

#[inline(always)]
pub fn debug_random() -> f32 {
    rand::random()
}

/// Uniformly distributed wrt the volume measure on the unit ball.
#[inline(always)]
pub fn random_in_unit_sphere<S: Simd>(r: Sample3D) -> Vec3<S> {
    let u = r.x * PI * 2.0;
    let v = (2.0 * r.y - 1.0).acos();
    let w = r.z.powf(1.0 / 3.0);
    Vec3::new(u.cos() * v.sin() * w, v.cos() * w, u.sin() * v.sin() * w)
}

/// Uniformly distributed wrt the surface area / solid angle measure.
#[inline(always)]
pub fn random_on_unit_sphere<S: Simd>(r: Sample2D) -> Vec3<S> {
    let Sample2D { x, y } = r;

    let phi = x * 2.0 * PI;
    let z = y * 2.0 - 1.0;
    let r = (1.0 - z * z).sqrt();

    let (s, c) = phi.sin_cos();

    Vec3::new(r * c, r * s, z)
}

/// Uniformly distributed wrt the area measure on the unit disk.
#[inline(always)]
pub fn random_in_unit_disk<S: Simd>(r: Sample2D) -> Vec3<S> {
    let u: f32 = r.x * PI * 2.0;
    let v: f32 = r.y.powf(1.0 / 2.0);
    Vec3::new(u.cos() * v, u.sin() * v, 0.0)
}

/// Cosine-weighted hemisphere direction. Uniform wrt projected solid angle.
#[inline(always)]
pub fn random_cosine_direction<S: Simd>(r: Sample2D) -> Vec3<S> {
    let Sample2D { x: u, y: v } = r;
    let z: f32 = (1.0 - v).sqrt();
    let phi: f32 = 2.0 * PI * u;
    let (mut y, mut x) = phi.sin_cos();
    let vsqrt = v.sqrt();
    x *= vsqrt;
    y *= vsqrt;
    Vec3::new(x, y, z)
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
    let r1 = r.x;
    let r2 = r.y;
    let z = 1.0 + r2 * ((1.0 - radius * radius / distance_squared).sqrt() - 1.0);
    let phi = 2.0 * PI * r1;
    let (mut y, mut x) = phi.sin_cos();
    let sqrt_1_z2 = (1.0 - z * z).sqrt();
    x *= sqrt_1_z2;
    y *= sqrt_1_z2;
    Vec3::new(x, y, z)
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
    }
}
