//! Forward-mode automatic differentiation for self-pdf'ing samplers.
//!
//! A warp `X = T(u)` from uniform samples `u ∈ [0,1)ⁿ` pushes the (unit) uniform
//! density to `p(X) = 1 / |volume scaling of T|`. Running the warp on dual-valued
//! inputs recovers the Jacobian of `T` as a byproduct, so the pdf falls out of
//! the *same* code that produces the sample and can never drift out of sync with
//! it.
//!
//! - [`Dual<N>`] is a scalar carrying its value plus `N` partial derivatives
//!   (one per consumed uniform dimension).
//! - [`SampleField`] is the small arithmetic/transcendental vocabulary the warps
//!   use, impl'd for plain `f32` (fast path, no derivatives) and for `Dual<N>`
//!   (pdf path). A warp written once against `SampleField` serves both.
//! - [`reciprocal_gram_det_2`] turns the `3×2` Jacobian of a 2-input warp into
//!   the pdf w.r.t. the induced surface measure via the Gram determinant
//!   `1/√det(JᵀJ)` (see `MEASURE_TYPE_TODOS.md` / the research plan).

use std::ops::{Add, Div, Mul, Neg, Sub};

/// A dual number: `val` plus `N` partial derivatives w.r.t. the `N` uniform
/// input dimensions. Forward-mode AD — arithmetic propagates value and
/// derivatives together.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Dual<const N: usize> {
    pub val: f32,
    pub eps: [f32; N],
}

impl<const N: usize> Dual<N> {
    /// A constant: value `v`, all partials zero.
    #[inline(always)]
    pub fn constant(v: f32) -> Self {
        Dual {
            val: v,
            eps: [0.0; N],
        }
    }

    /// The `i`-th independent input variable with value `v`: `∂/∂xᵢ = 1`, the
    /// rest zero. This is how a uniform sample enters a warp on the pdf path.
    #[inline(always)]
    pub fn variable(v: f32, i: usize) -> Self {
        let mut eps = [0.0; N];
        eps[i] = 1.0;
        Dual { val: v, eps }
    }
}

impl<const N: usize> Add for Dual<N> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        let mut eps = self.eps;
        for i in 0..N {
            eps[i] += rhs.eps[i];
        }
        Dual {
            val: self.val + rhs.val,
            eps,
        }
    }
}

impl<const N: usize> Sub for Dual<N> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        let mut eps = self.eps;
        for i in 0..N {
            eps[i] -= rhs.eps[i];
        }
        Dual {
            val: self.val - rhs.val,
            eps,
        }
    }
}

impl<const N: usize> Mul for Dual<N> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // product rule: (ab)' = a'b + ab'
        let mut eps = [0.0; N];
        for i in 0..N {
            eps[i] = self.eps[i] * rhs.val + self.val * rhs.eps[i];
        }
        Dual {
            val: self.val * rhs.val,
            eps,
        }
    }
}

impl<const N: usize> Div for Dual<N> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        // quotient rule: (a/b)' = (a'b - ab') / b²
        let inv = 1.0 / rhs.val;
        let inv2 = inv * inv;
        let mut eps = [0.0; N];
        for i in 0..N {
            eps[i] = (self.eps[i] * rhs.val - self.val * rhs.eps[i]) * inv2;
        }
        Dual {
            val: self.val * inv,
            eps,
        }
    }
}

impl<const N: usize> Neg for Dual<N> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        let mut eps = self.eps;
        for i in 0..N {
            eps[i] = -eps[i];
        }
        Dual {
            val: -self.val,
            eps,
        }
    }
}

/// The arithmetic + transcendental vocabulary the warp routines in
/// [`crate::random`] use. Impl'd for `f32` (value-only fast path) and `Dual<N>`
/// (value + derivatives). A warp written generically over `SampleField` can be
/// instantiated either way: `f32` for sampling, `Dual<N>` to also get the pdf.
pub trait SampleField:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// A literal constant (zero derivative on the dual path).
    fn constant(v: f32) -> Self;
    /// Square root.
    fn sqrt(self) -> Self;
    /// `(sin, cos)` together (matches `f32::sin_cos`).
    fn sin_cos(self) -> (Self, Self);
    /// Inverse cosine.
    fn acos(self) -> Self;
    /// `self` raised to a constant power.
    fn powf(self, p: f32) -> Self;
    /// The underlying value, for building the concrete sample at the boundary.
    fn value(self) -> f32;
}

impl SampleField for f32 {
    #[inline(always)]
    fn constant(v: f32) -> Self {
        v
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    #[inline(always)]
    fn sin_cos(self) -> (Self, Self) {
        f32::sin_cos(self)
    }
    #[inline(always)]
    fn acos(self) -> Self {
        f32::acos(self)
    }
    #[inline(always)]
    fn powf(self, p: f32) -> Self {
        f32::powf(self, p)
    }
    #[inline(always)]
    fn value(self) -> f32 {
        self
    }
}

impl<const N: usize> SampleField for Dual<N> {
    #[inline(always)]
    fn constant(v: f32) -> Self {
        Dual::constant(v)
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        // d/dx √x = 1/(2√x)
        let s = self.val.sqrt();
        let d = 0.5 / s;
        let mut eps = self.eps;
        for i in 0..N {
            eps[i] *= d;
        }
        Dual { val: s, eps }
    }
    #[inline(always)]
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.val.sin_cos();
        let mut sin_eps = self.eps;
        let mut cos_eps = self.eps;
        for i in 0..N {
            sin_eps[i] *= c; // d sin = cos · dx
            cos_eps[i] *= -s; // d cos = -sin · dx
        }
        (
            Dual { val: s, eps: sin_eps },
            Dual { val: c, eps: cos_eps },
        )
    }
    #[inline(always)]
    fn acos(self) -> Self {
        // d/dx acos(x) = -1/√(1-x²)
        let d = -1.0 / (1.0 - self.val * self.val).sqrt();
        let mut eps = self.eps;
        for i in 0..N {
            eps[i] *= d;
        }
        Dual {
            val: self.val.acos(),
            eps,
        }
    }
    #[inline(always)]
    fn powf(self, p: f32) -> Self {
        // d/dx xᵖ = p·xᵖ⁻¹
        let d = p * self.val.powf(p - 1.0);
        let mut eps = self.eps;
        for i in 0..N {
            eps[i] *= d;
        }
        Dual {
            val: self.val.powf(p),
            eps,
        }
    }
    #[inline(always)]
    fn value(self) -> f32 {
        self.val
    }
}

/// Reciprocal Gram determinant of the `3×2` Jacobian of a two-input warp:
/// `1 / √det(JᵀJ)`. This is the density w.r.t. the surface measure induced on
/// the warp's image — solid angle for a warp onto the unit sphere, area for a
/// planar (`z`-constant) warp, since the zero `z`-row makes the Gram determinant
/// equal the square of the in-plane `2×2` determinant.
#[inline(always)]
pub fn reciprocal_gram_det_2(out: &[Dual<2>; 3]) -> f32 {
    // J is 3×2 (rows = output components, cols = ∂/∂u, ∂/∂v). G = JᵀJ is 2×2.
    let mut g00 = 0.0f32;
    let mut g01 = 0.0f32;
    let mut g11 = 0.0f32;
    for o in out {
        g00 += o.eps[0] * o.eps[0];
        g01 += o.eps[0] * o.eps[1];
        g11 += o.eps[1] * o.eps[1];
    }
    let det = g00 * g11 - g01 * g01;
    1.0 / det.sqrt()
}

/// Reciprocal absolute determinant of the `3×3` Jacobian of a three-input warp:
/// `1 / |det J|`. For a full-dimensional warp (`ℝ³ → ℝ³`) the Gram determinant
/// `√det(JᵀJ)` reduces to `|det J|`, so this is the density w.r.t. the volume
/// measure on the warp's image.
#[inline(always)]
pub fn reciprocal_det_3(out: &[Dual<3>; 3]) -> f32 {
    // J[row][col] = ∂out[row]/∂uᶜᵒˡ
    let j = [out[0].eps, out[1].eps, out[2].eps];
    let det = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
        - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
        + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
    1.0 / det.abs()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn constant_has_zero_derivative() {
        let c = Dual::<2>::constant(3.0);
        assert_eq!(c.val, 3.0);
        assert_eq!(c.eps, [0.0, 0.0]);
    }

    #[test]
    fn product_rule() {
        // f = x*y at (x,y)=(2,3): f=6, ∂f/∂x=y=3, ∂f/∂y=x=2
        let x = Dual::<2>::variable(2.0, 0);
        let y = Dual::<2>::variable(3.0, 1);
        let f = x * y;
        assert_eq!(f.val, 6.0);
        assert_eq!(f.eps, [3.0, 2.0]);
    }

    #[test]
    fn quotient_rule() {
        // f = x/y at (4,2): f=2, ∂/∂x=1/y=0.5, ∂/∂y=-x/y²=-1
        let x = Dual::<2>::variable(4.0, 0);
        let y = Dual::<2>::variable(2.0, 1);
        let f = x / y;
        assert_eq!(f.val, 2.0);
        assert!((f.eps[0] - 0.5).abs() < 1e-6);
        assert!((f.eps[1] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn sqrt_derivative() {
        // d/dx √x at x=4 is 1/(2·2)=0.25
        let x = Dual::<1>::variable(4.0, 0);
        let f = x.sqrt();
        assert!((f.val - 2.0).abs() < 1e-6);
        assert!((f.eps[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn acos_derivative() {
        // d/dx acos(x) at x=0 is -1/√1 = -1
        let x = Dual::<1>::variable(0.0, 0);
        let f = x.acos();
        assert!((f.val - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        assert!((f.eps[0] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn powf_derivative() {
        // f = x^3 at x=2: f=8, f'=3·x²=12
        let x = Dual::<1>::variable(2.0, 0);
        let f = x.powf(3.0);
        assert!((f.val - 8.0).abs() < 1e-5);
        assert!((f.eps[0] - 12.0).abs() < 1e-4);
    }

    #[test]
    fn det3_of_diagonal() {
        // a warp with diagonal Jacobian diag(2,3,4) → det = 24, reciprocal 1/24
        let out = [
            Dual::<3> { val: 0.0, eps: [2.0, 0.0, 0.0] },
            Dual::<3> { val: 0.0, eps: [0.0, 3.0, 0.0] },
            Dual::<3> { val: 0.0, eps: [0.0, 0.0, 4.0] },
        ];
        assert!((reciprocal_det_3(&out) - 1.0 / 24.0).abs() < 1e-6);
    }

    #[test]
    fn sin_cos_derivative() {
        // at x=0: sin=0 (d=cos=1), cos=1 (d=-sin=0)
        let x = Dual::<1>::variable(0.0, 0);
        let (s, c) = x.sin_cos();
        assert!((s.val).abs() < 1e-6);
        assert!((s.eps[0] - 1.0).abs() < 1e-6);
        assert!((c.val - 1.0).abs() < 1e-6);
        assert!((c.eps[0]).abs() < 1e-6);
    }
}
