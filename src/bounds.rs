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
