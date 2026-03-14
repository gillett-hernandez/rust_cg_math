use crate::prelude::*;

#[derive(Copy, Clone, Debug)]
pub struct Ray {
    pub origin: Point3,
    pub direction: Vec3,
    pub time: f32,
    pub tmax: f32,
}

impl Ray {
    pub const fn new(origin: Point3, direction: Vec3) -> Self {
        Ray {
            origin,
            direction,
            time: 0.0,
            tmax: INFINITY,
        }
    }

    pub const fn new_with_time(origin: Point3, direction: Vec3, time: f32) -> Self {
        Ray {
            origin,
            direction,
            time,
            tmax: INFINITY,
        }
    }
    pub const fn new_with_time_and_tmax(
        origin: Point3,
        direction: Vec3,
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
    pub fn with_tmax(mut self, tmax: f32) -> Self {
        self.tmax = tmax;
        self
    }
    pub fn at_time(mut self, time: f32) -> Self {
        self.origin = self.point_at_parameter(time);
        self
    }
    pub fn point_at_parameter(self, time: f32) -> Point3 {
        self.origin + self.direction * time
    }
}

impl Default for Ray {
    fn default() -> Self {
        Ray::new(Point3::default(), Vec3::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn arb_vec3() -> impl Strategy<Value = Vec3> {
        (-1e4f32..1e4, -1e4f32..1e4, -1e4f32..1e4)
            .prop_map(|(x, y, z)| Vec3::new(x, y, z))
    }

    fn arb_point3() -> impl Strategy<Value = Point3> {
        (-1e4f32..1e4, -1e4f32..1e4, -1e4f32..1e4)
            .prop_map(|(x, y, z)| Point3::new(x, y, z))
    }

    fn arb_direction() -> impl Strategy<Value = Vec3> {
        arb_vec3().prop_filter("nonzero", |v| v.norm() > 1e-6)
            .prop_map(|v| v.normalized())
    }

    #[test]
    fn test_default_ray() {
        let r = Ray::default();
        assert_eq!(r.origin, Point3::default());
        assert_eq!(r.time, 0.0);
        assert_eq!(r.tmax, INFINITY);
    }

    proptest! {
        #[test]
        fn point_at_zero_is_origin(origin in arb_point3(), dir in arb_direction()) {
            let ray = Ray::new(origin, dir);
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
            let ray = Ray::new(origin, dir);
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
            let ray = Ray::new_with_time(origin, dir, time).with_tmax(tmax);
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
            let ray = Ray::new_with_time(origin, dir, time);
            prop_assert_eq!(ray.time, time);
            prop_assert_eq!(ray.tmax, INFINITY);
        }
    }
}
