use crate::prelude::*;
use thermite::simd::Simd;
use thermite::register::LinAlg3Register;

pub struct Ray<S: Simd> {
    pub origin: Point3<S>,
    pub direction: Vec3<S>,
    pub time: f32,
    pub tmax: f32,
}

impl<S: Simd> Copy for Ray<S> {}
impl<S: Simd> Clone for Ray<S> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<S: Simd> std::fmt::Debug for Ray<S> {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ray")
            .field("origin", &self.origin)
            .field("direction", &self.direction)
            .field("time", &self.time)
            .field("tmax", &self.tmax)
            .finish()
    }
}

impl<S: Simd> Ray<S> {
    #[inline(always)]
    pub const fn new(origin: Point3<S>, direction: Vec3<S>) -> Self {
        Ray {
            origin,
            direction,
            time: 0.0,
            tmax: INFINITY,
        }
    }

    #[inline(always)]
    pub const fn new_with_time(origin: Point3<S>, direction: Vec3<S>, time: f32) -> Self {
        Ray {
            origin,
            direction,
            time,
            tmax: INFINITY,
        }
    }
    #[inline(always)]
    pub const fn new_with_time_and_tmax(
        origin: Point3<S>,
        direction: Vec3<S>,
        time: f32,
        tmax: f32,
    ) -> Self {
        Ray {
            origin,
            direction,
            time,
            tmax,
        }
    }
    #[inline(always)]
    pub fn with_tmax(mut self, tmax: f32) -> Self {
        self.tmax = tmax;
        self
    }
}

impl<S: Simd> Ray<S>
where
    S::f32x4: LinAlg3Register,
{
    #[inline(always)]
    pub fn at_time(mut self, time: f32) -> Self {
        self.origin = self.point_at_parameter(time);
        self
    }
    #[inline(always)]
    pub fn point_at_parameter(self, time: f32) -> Point3<S> {
        self.origin + self.direction * time
    }
}

impl<S: Simd> Default for Ray<S> {
    #[inline(always)]
    fn default() -> Self {
        Ray::new(Point3::default(), Vec3::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    type TestS = thermite::backend::scalar::Scalar;
    type V3 = Vec3<TestS>;
    type P3 = Point3<TestS>;
    type R = Ray<TestS>;

    fn arb_vec3() -> impl Strategy<Value = V3> {
        (-1e4f32..1e4, -1e4f32..1e4, -1e4f32..1e4).prop_map(|(x, y, z)| V3::new(x, y, z))
    }

    fn arb_point3() -> impl Strategy<Value = P3> {
        (-1e4f32..1e4, -1e4f32..1e4, -1e4f32..1e4).prop_map(|(x, y, z)| P3::new(x, y, z))
    }

    fn arb_direction() -> impl Strategy<Value = V3> {
        arb_vec3()
            .prop_filter("nonzero", |v| v.norm() > 1e-6)
            .prop_map(|v| v.normalized())
    }

    #[test]
    fn test_default_ray() {
        let r = R::default();
        assert_eq!(r.origin, P3::default());
        assert_eq!(r.time, 0.0);
        assert_eq!(r.tmax, INFINITY);
    }

    #[test]
    fn test_debug_and_clone() {
        let r = R::new(P3::new(1.0, 2.0, 3.0), V3::x_axis());
        let s = format!("{:?}", r);
        assert!(s.contains("Ray"));
        let c = r.clone();
        assert_eq!(c.origin, r.origin);
        assert_eq!(c.tmax, r.tmax);
    }

    #[test]
    fn test_new_with_time_and_tmax() {
        let r = R::new_with_time_and_tmax(P3::origin(), V3::z_axis(), 1.5, 42.0);
        assert_eq!(r.time, 1.5);
        assert_eq!(r.tmax, 42.0);
        assert_eq!(r.origin, P3::origin());
    }

    #[test]
    fn test_at_time_advances_origin() {
        let r = R::new(P3::origin(), V3::x_axis());
        let advanced = r.at_time(5.0);
        let diff = (advanced.origin - P3::new(5.0, 0.0, 0.0)).norm();
        assert!(diff < 1e-5, "origin not advanced: {:?}", advanced.origin);
    }

    proptest! {
        #[test]
        fn point_at_zero_is_origin(origin in arb_point3(), dir in arb_direction()) {
            let ray = R::new(origin, dir);
            let p = ray.point_at_parameter(0.0);
            let diff = (p - origin).norm();
            prop_assert!(diff < 1e-4, "ray(0) != origin, diff={}", diff);
        }

        #[test]
        fn point_at_parameter_linearity(
            origin in arb_point3(),
            dir in arb_direction(),
            t in -100.0f32..100.0
        ) {
            let ray = R::new(origin, dir);
            let p = ray.point_at_parameter(t);
            let expected = origin + dir * t;
            let diff = (p - expected).norm();
            prop_assert!(diff < 1e-2, "ray(t) != origin + t*dir, diff={}", diff);
        }

        #[test]
        fn with_tmax_preserves_other_fields(
            origin in arb_point3(),
            dir in arb_direction(),
            time in 0.0f32..10.0,
            tmax in 0.1f32..1000.0
        ) {
            let ray = R::new_with_time(origin, dir, time).with_tmax(tmax);
            prop_assert_eq!(ray.tmax, tmax);
            prop_assert_eq!(ray.time, time);
            let diff = (ray.origin - origin).norm();
            prop_assert!(diff < 1e-6);
        }

        #[test]
        fn new_with_time_sets_time(
            origin in arb_point3(),
            dir in arb_direction(),
            time in 0.0f32..10.0
        ) {
            let ray = R::new_with_time(origin, dir, time);
            prop_assert_eq!(ray.time, time);
            prop_assert_eq!(ray.tmax, INFINITY);
        }
    }
}
