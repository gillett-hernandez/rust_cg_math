use crate::prelude::*;

pub fn debug_random() -> f32 {
    // uses thread local rng
    rand::random()
}

/// theoretically, this is uniformly distributed wrt the volume measure.
/// testable by confirming that the monte carlo estimate with these samples is equal to the volume of a sphere,  4/3 * pi
pub fn random_in_unit_sphere(r: Sample3D) -> Vec3 {
    let u = r.x * PI * 2.0;
    let v = (2.0 * r.y - 1.0).acos();
    let w = r.z.powf(1.0 / 3.0);
    Vec3::new(u.cos() * v.sin() * w, v.cos() * w, u.sin() * v.sin() * w)
}

/// theoretically, this is uniformly distributed wrt the surface area / solid angle measure
pub fn random_on_unit_sphere(r: Sample2D) -> Vec3 {
    let Sample2D { x, y } = r;

    let phi = x * 2.0 * PI;
    let z = y * 2.0 - 1.0;
    let r = (1.0 - z * z).sqrt();

    let (s, c) = phi.sin_cos();

    Vec3::new(r * c, r * s, z)
}

/// theoretically, this is uniformly distributed wrt the area measure
pub fn random_in_unit_disk(r: Sample2D) -> Vec3 {
    let u: f32 = r.x * PI * 2.0;
    let v: f32 = r.y.powf(1.0 / 2.0);
    Vec3::new(u.cos() * v, u.sin() * v, 0.0)
}

/// returns a random direction (unit vector) in the positive Z hemisphere, distributed according to the cosine distribution.
/// theoretically, this is uniformly distributed wrt the projected solid angle measure.
pub fn random_cosine_direction(r: Sample2D) -> Vec3 {
    let Sample2D { x: u, y: v } = r;
    let z: f32 = (1.0 - v).sqrt();
    let phi: f32 = 2.0 * PI * u;
    let (mut y, mut x) = phi.sin_cos();
    let vsqrt = v.sqrt();
    x *= vsqrt;
    y *= vsqrt;
    Vec3::new(x, y, z)
}

pub fn weighted_cosine_direction(r: Sample2D, weight: f32) -> Vec3 {
    let Sample2D { x: u, y: v } = r;
    let z: f32 = weight * (1.0 - v).sqrt();
    let phi: f32 = 2.0 * PI * u;
    let (mut y, mut x) = phi.sin_cos();
    x *= v.sqrt();
    y *= v.sqrt();
    Vec3::new(x, y, z).normalized()
}

pub fn random_to_sphere(r: Sample2D, radius: f32, distance_squared: f32) -> Vec3 {
    let r1 = r.x;
    let r2 = r.y;
    let z = 1.0 + r2 * ((1.0 - radius * radius / distance_squared).sqrt() - 1.0);
    let phi = 2.0 * PI * r1;
    let (mut y, mut x) = phi.sin_cos();
    let sqrt_1_z2 = (1.0 - z * z).sqrt();
    x *= sqrt_1_z2;
    y *= sqrt_1_z2;
    return Vec3::new(x, y, z);
}
