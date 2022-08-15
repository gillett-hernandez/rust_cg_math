
use crate::prelude::*;

use packed_simd::f32x4;

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign};

// #[derive(Copy, Clone, Debug, PartialEq)]
// pub struct HeroEnergy(pub f32x4);

// impl HeroEnergy {
//     pub fn new(energy: f32x4) -> Self {
//         HeroEnergy { 0: energy }
//     }
//     pub const ZERO: HeroEnergy = HeroEnergy {
//         0: f32x4::splat(0.0),
//     };
//     pub const ONE: HeroEnergy = HeroEnergy {
//         0: f32x4::splat(1.0),
//     };
//     pub fn is_nan(&self) -> bool {
//         self.0.is_nan().any()
//     }
// }
// impl Add for HeroEnergy {
//     type Output = HeroEnergy;
//     fn add(self, rhs: HeroEnergy) -> Self::Output {
//         HeroEnergy::new(self.0 + rhs.0)
//     }
// }
// impl AddAssign for HeroEnergy {
//     fn add_assign(&mut self, rhs: HeroEnergy) {
//         self.0 += rhs.0;
//     }
// }

// impl Mul<f32> for HeroEnergy {
//     type Output = HeroEnergy;
//     fn mul(self, rhs: f32) -> Self::Output {
//         HeroEnergy::new(self.0 * rhs)
//     }
// }
// impl Mul<HeroEnergy> for f32 {
//     type Output = HeroEnergy;
//     fn mul(self, rhs: HeroEnergy) -> Self::Output {
//         HeroEnergy::new(self * rhs.0)
//     }
// }

// impl Mul for HeroEnergy {
//     type Output = HeroEnergy;
//     fn mul(self, rhs: HeroEnergy) -> Self::Output {
//         HeroEnergy::new(self.0 * rhs.0)
//     }
// }

// impl MulAssign for HeroEnergy {
//     fn mul_assign(&mut self, other: HeroEnergy) {
//         self.0 = self.0 * other.0
//     }
// }

// impl MulAssign<f32> for HeroEnergy {
//     fn mul_assign(&mut self, other: f32) {
//         self.0 = self.0 * other
//     }
// }

// impl Div<f32> for HeroEnergy {
//     type Output = HeroEnergy;
//     fn div(self, rhs: f32) -> Self::Output {
//         HeroEnergy::new(self.0 / rhs)
//     }
// }

// impl From<f32> for HeroEnergy {
//     fn from(value: f32) -> Self {
//         HeroEnergy::new(f32x4::splat(value))
//     }
// }

// impl From<f32x4> for HeroEnergy {
//     fn from(value: f32x4) -> Self {
//         HeroEnergy::new(value)
//     }
// }

// #[derive(Copy, Clone, Debug)]
// pub struct HeroWavelength {
//     pub lambda: f32x4,
//     pub energy: HeroEnergy,
// }

// impl HeroWavelength {
//     pub const fn new(lambda: f32x4, energy: HeroEnergy) -> HeroWavelength {
//         HeroWavelength { lambda, energy }
//     }

//     pub fn new_from_range(x: f32, bounds: Bounds1D) -> Self {
//         let hero = x * bounds.span();
//         let delta = bounds.span() / 4.0;
//         let mult = f32x4::new(0.0, 1.0, 2.0, 3.0);
//         let wavelengths = bounds.lower + (hero + mult * delta);
//         let sub: f32x4 = wavelengths
//             .gt(f32x4::splat(bounds.upper))
//             .select(f32x4::splat(bounds.span()), f32x4::splat(0.0));

//         HeroWavelength::new(wavelengths - sub, HeroEnergy::ZERO)
//     }

//     pub fn with_energy(&self, energy: HeroEnergy) -> Self {
//         HeroWavelength::new(self.lambda, energy)
//     }

//     pub fn replace_energy(&self, energy: f32x4) -> Self {
//         self.with_energy(HeroEnergy::new(energy))
//     }

//     pub const BLACK: HeroWavelength = HeroWavelength::new(f32x4::splat(0.0), HeroEnergy::ZERO);
// }

// impl Mul<f32> for HeroWavelength {
//     type Output = HeroWavelength;
//     fn mul(self, other: f32) -> HeroWavelength {
//         self.with_energy(self.energy * other)
//     }
// }

// impl Mul<HeroWavelength> for f32 {
//     type Output = HeroWavelength;
//     fn mul(self, other: HeroWavelength) -> HeroWavelength {
//         other.with_energy(self * other.energy)
//     }
// }

// impl Mul<XYZColor> for HeroWavelength {
//     type Output = HeroWavelength;
//     fn mul(self, _xyz: XYZColor) -> HeroWavelength {
//         // let lambda = other.wavelength;
//         // let other_as_color: XYZColor = other.into();
//         // other_as_color gives us the x y and z values for other
//         // self.with_energy(self.energy * xyz.y())
//         unimplemented!()
//     }
// }

// impl Div<f32> for HeroWavelength {
//     type Output = HeroWavelength;
//     fn div(self, other: f32) -> HeroWavelength {
//         self.with_energy(self.energy / other)
//     }
// }

// impl DivAssign<f32> for HeroWavelength {
//     fn div_assign(&mut self, other: f32) {
//         self.energy = self.energy / other;
//     }
// }

// impl Mul<HeroEnergy> for HeroWavelength {
//     type Output = HeroWavelength;
//     fn mul(self, rhs: HeroEnergy) -> Self::Output {
//         self.with_energy(self.energy * rhs)
//     }
// }

// impl Mul<HeroWavelength> for HeroEnergy {
//     type Output = HeroWavelength;
//     fn mul(self, rhs: HeroWavelength) -> Self::Output {
//         rhs.with_energy(self * rhs.energy)
//     }
// }
