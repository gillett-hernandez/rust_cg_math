use crate::prelude::*;

// use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign};



// #[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
// pub struct SingleEnergy(pub f32);

// impl SingleEnergy {
//     pub fn new(energy: f32) -> Self {
//         SingleEnergy { 0: energy }
//     }
//     pub const ZERO: SingleEnergy = SingleEnergy { 0: 0.0 };
//     pub const ONE: SingleEnergy = SingleEnergy { 0: 1.0 };
//     pub fn is_nan(&self) -> bool {
//         self.0.is_nan()
//     }
// }
// impl Add for SingleEnergy {
//     type Output = SingleEnergy;
//     fn add(self, rhs: SingleEnergy) -> Self::Output {
//         SingleEnergy::new(self.0 + rhs.0)
//     }
// }
// impl AddAssign for SingleEnergy {
//     fn add_assign(&mut self, rhs: SingleEnergy) {
//         self.0 += rhs.0;
//     }
// }

// impl Mul<f32> for SingleEnergy {
//     type Output = SingleEnergy;
//     fn mul(self, rhs: f32) -> Self::Output {
//         SingleEnergy::new(self.0 * rhs)
//     }
// }
// impl Mul<SingleEnergy> for f32 {
//     type Output = SingleEnergy;
//     fn mul(self, rhs: SingleEnergy) -> Self::Output {
//         SingleEnergy::new(self * rhs.0)
//     }
// }

// impl Mul for SingleEnergy {
//     type Output = SingleEnergy;
//     fn mul(self, rhs: SingleEnergy) -> Self::Output {
//         SingleEnergy::new(self.0 * rhs.0)
//     }
// }

// impl MulAssign for SingleEnergy {
//     fn mul_assign(&mut self, other: SingleEnergy) {
//         self.0 = self.0 * other.0
//     }
// }

// impl MulAssign<f32> for SingleEnergy {
//     fn mul_assign(&mut self, other: f32) {
//         self.0 = self.0 * other
//     }
// }

// impl Div<f32> for SingleEnergy {
//     type Output = SingleEnergy;
//     fn div(self, rhs: f32) -> Self::Output {
//         SingleEnergy::new(self.0 / rhs)
//     }
// }

// impl From<f32> for SingleEnergy {
//     fn from(value: f32) -> Self {
//         SingleEnergy::new(value)
//     }
// }

// #[derive(Copy, Clone, Debug)]
// pub struct SingleWavelength {
//     pub lambda: f32,
//     pub energy: SingleEnergy,
// }

// impl SingleWavelength {
//     pub const fn new(lambda: f32, energy: SingleEnergy) -> SingleWavelength {
//         SingleWavelength { lambda, energy }
//     }

//     pub fn new_from_range(x: f32, bounds: Bounds1D) -> Self {
//         SingleWavelength::new(bounds.lower + x * bounds.span(), SingleEnergy::ZERO)
//     }

//     pub fn with_energy(&self, energy: SingleEnergy) -> Self {
//         SingleWavelength::new(self.lambda, energy)
//     }

//     pub fn replace_energy(&self, energy: f32) -> Self {
//         self.with_energy(SingleEnergy::new(energy))
//     }

//     pub const BLACK: SingleWavelength = SingleWavelength::new(0.0, SingleEnergy::ZERO);
// }

// impl Mul<f32> for SingleWavelength {
//     type Output = SingleWavelength;
//     fn mul(self, other: f32) -> SingleWavelength {
//         self.with_energy(self.energy * other)
//     }
// }

// impl Mul<SingleWavelength> for f32 {
//     type Output = SingleWavelength;
//     fn mul(self, other: SingleWavelength) -> SingleWavelength {
//         other.with_energy(self * other.energy)
//     }
// }

// impl Mul<XYZColor> for SingleWavelength {
//     type Output = SingleWavelength;
//     fn mul(self, _xyz: XYZColor) -> SingleWavelength {
//         // let lambda = other.wavelength;
//         // let other_as_color: XYZColor = other.into();
//         // other_as_color gives us the x y and z values for other
//         // self.with_energy(self.energy * xyz.y())
//         unimplemented!()
//     }
// }

// impl Div<f32> for SingleWavelength {
//     type Output = SingleWavelength;
//     fn div(self, other: f32) -> SingleWavelength {
//         self.with_energy(self.energy / other)
//     }
// }

// impl DivAssign<f32> for SingleWavelength {
//     fn div_assign(&mut self, other: f32) {
//         self.energy = self.energy / other;
//     }
// }

// impl Mul<SingleEnergy> for SingleWavelength {
//     type Output = SingleWavelength;
//     fn mul(self, rhs: SingleEnergy) -> Self::Output {
//         self.with_energy(self.energy * rhs)
//     }
// }

// impl Mul<SingleWavelength> for SingleEnergy {
//     type Output = SingleWavelength;
//     fn mul(self, rhs: SingleWavelength) -> Self::Output {
//         rhs.with_energy(self * rhs.energy)
//     }
// }
