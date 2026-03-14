#[cfg(feature = "deepsize")]
use deepsize::DeepSizeOf;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[cfg_attr(feature = "deepsize", derive(DeepSizeOf))]
pub struct Bounds1D {
    pub lower: f32,
    pub upper: f32,
}

impl Bounds1D {
    pub const fn new(lower: f32, upper: f32) -> Self {
        Bounds1D { lower, upper }
    }
    pub fn span(&self) -> f32 {
        self.upper - self.lower
    }

    pub fn lerp(&self, t: f32) -> f32 {
        t * self.span() + self.lower
    }

    pub fn contains(&self, value: &f32) -> bool {
        &self.lower <= value && value < &self.upper
    }
    pub fn intersection(&self, other: Self) -> Self {
        Bounds1D::new(self.lower.max(other.lower), self.upper.min(other.upper))
    }

    pub fn union(&self, other: Self) -> Self {
        Bounds1D::new(self.lower.min(other.lower), self.upper.max(other.upper))
    }
    pub fn sample(&self, x: f32) -> f32 {
        x * self.span() + self.lower
    }
}

#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[cfg_attr(feature = "deepsize", derive(DeepSizeOf))]
pub struct Bounds2D {
    pub x: Bounds1D,
    pub y: Bounds1D,
}

impl Bounds2D {
    pub const fn new(x: Bounds1D, y: Bounds1D) -> Self {
        Bounds2D { x, y }
    }
    pub fn area(&self) -> f32 {
        self.x.span() * self.y.span()
    }

    pub fn contains(&self, value: (f32, f32)) -> bool {
        self.x.contains(&value.0) && self.y.contains(&value.1)
    }
    pub fn intersection(&self, other: Self) -> Self {
        Bounds2D::new(self.x.intersection(other.x), self.y.intersection(other.y))
    }

    pub fn union(&self, other: Self) -> Self {
        Bounds2D::new(self.x.union(other.x), self.y.union(other.y))
    }
    pub fn sample(&self, x: f32, y: f32) -> (f32, f32) {
        (self.x.sample(x), self.y.sample(y))
    }
}

impl From<(f32, f32)> for Bounds1D {
    fn from(mut bounds: (f32, f32)) -> Self {
        // swap if in the wrong order
        bounds = if bounds.0 > bounds.1 {
            (bounds.1, bounds.0)
        } else {
            bounds
        };
        Self::new(bounds.0, bounds.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn arb_bounds1d() -> impl Strategy<Value = Bounds1D> {
        (-1e3f32..1e3, -1e3f32..1e3)
            .prop_filter("lower < upper", |(a, b)| (b - a).abs() > 1e-6)
            .prop_map(|(a, b)| {
                if a < b {
                    Bounds1D::new(a, b)
                } else {
                    Bounds1D::new(b, a)
                }
            })
    }

    fn arb_bounds2d() -> impl Strategy<Value = Bounds2D> {
        (arb_bounds1d(), arb_bounds1d()).prop_map(|(x, y)| Bounds2D::new(x, y))
    }

    proptest! {
        #[test]
        fn span_is_non_negative(b in arb_bounds1d()) {
            prop_assert!(b.span() >= 0.0, "span={}", b.span());
        }

        #[test]
        fn lerp_endpoints(b in arb_bounds1d()) {
            let at_zero = b.lerp(0.0);
            let at_one = b.lerp(1.0);
            prop_assert!((at_zero - b.lower).abs() < 1e-4, "lerp(0)={}, lower={}", at_zero, b.lower);
            prop_assert!((at_one - b.upper).abs() < 1e-4, "lerp(1)={}, upper={}", at_one, b.upper);
        }

        #[test]
        fn lerp_in_bounds(b in arb_bounds1d(), t in 0.001f32..0.999) {
            let val = b.lerp(t);
            prop_assert!(b.contains(&val), "lerp({})={} not in [{}, {})", t, val, b.lower, b.upper);
        }

        #[test]
        fn contains_lower_not_upper(b in arb_bounds1d()) {
            prop_assert!(b.contains(&b.lower), "should contain lower");
            prop_assert!(!b.contains(&b.upper), "should not contain upper (half-open)");
        }

        #[test]
        fn union_commutative(a in arb_bounds1d(), b in arb_bounds1d()) {
            let ab = a.union(b);
            let ba = b.union(a);
            prop_assert!((ab.lower - ba.lower).abs() < 1e-6);
            prop_assert!((ab.upper - ba.upper).abs() < 1e-6);
        }

        #[test]
        fn union_contains_both(a in arb_bounds1d(), b in arb_bounds1d()) {
            let u = a.union(b);
            prop_assert!(u.lower <= a.lower && u.upper >= a.upper);
            prop_assert!(u.lower <= b.lower && u.upper >= b.upper);
        }

        #[test]
        fn intersection_commutative(a in arb_bounds1d(), b in arb_bounds1d()) {
            let ab = a.intersection(b);
            let ba = b.intersection(a);
            prop_assert!((ab.lower - ba.lower).abs() < 1e-6);
            prop_assert!((ab.upper - ba.upper).abs() < 1e-6);
        }

        #[test]
        fn intersection_within_both(a in arb_bounds1d(), b in arb_bounds1d()) {
            let isect = a.intersection(b);
            // intersection lower >= max(a.lower, b.lower)
            // intersection upper <= min(a.upper, b.upper)
            prop_assert!(isect.lower >= a.lower.min(b.lower));
            prop_assert!(isect.upper <= a.upper.max(b.upper));
        }

        #[test]
        fn sample_equals_lerp(b in arb_bounds1d(), t in 0.0f32..1.0) {
            let sampled = b.sample(t);
            let lerped = b.lerp(t);
            prop_assert!((sampled - lerped).abs() < 1e-6);
        }

        #[test]
        fn from_tuple_orders_correctly(a in -1e3f32..1e3, b in -1e3f32..1e3) {
            let bounds = Bounds1D::from((a, b));
            prop_assert!(bounds.lower <= bounds.upper);
        }

        // Bounds2D tests
        #[test]
        fn area_equals_product_of_spans(b in arb_bounds2d()) {
            let area = b.area();
            let expected = b.x.span() * b.y.span();
            prop_assert!((area - expected).abs() < 1e-2, "area={}, expected={}", area, expected);
        }

        #[test]
        fn bounds2d_contains_interior(b in arb_bounds2d()) {
            let mid_x = b.x.lerp(0.5);
            let mid_y = b.y.lerp(0.5);
            prop_assert!(b.contains((mid_x, mid_y)), "midpoint not contained");
        }

        #[test]
        fn bounds2d_union_commutative(a in arb_bounds2d(), b in arb_bounds2d()) {
            let ab = a.union(b);
            let ba = b.union(a);
            prop_assert!((ab.x.lower - ba.x.lower).abs() < 1e-6);
            prop_assert!((ab.y.lower - ba.y.lower).abs() < 1e-6);
        }
    }
}
