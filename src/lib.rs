#[macro_use]
extern crate packed_simd;

mod bounds;
mod color;
pub mod curves;
mod misc;
mod point;
mod random;
mod sample;
pub mod spectral;
mod tangent_frame;
mod transform;
mod vec;

pub use bounds::*;
pub use color::*;
pub use misc::*;
pub use point::Point3;
pub use random::*;
pub use sample::*;
pub use spectral::{SingleEnergy, SingleWavelength, SpectralPowerDistributionFunction};

pub use curves::{Curve, CurveWithCDF};

pub use tangent_frame::TangentFrame;
pub use transform::*;
pub use vec::{Axis, Vec3};

pub use packed_simd::f32x4;

pub use std::f32::consts::PI;
pub use std::f32::INFINITY;

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Copy, Clone, PartialEq, Debug)]
pub enum Sidedness {
    Forward,
    Reverse,
    Dual,
}
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct PDF(pub f32);
impl PDF {
    pub fn is_nan(&self) -> bool {
        self.0.is_nan()
    }
}

impl From<f32> for PDF {
    fn from(val: f32) -> Self {
        PDF(val)
    }
}

impl From<PDF> for f32 {
    fn from(val: PDF) -> Self {
        val.0
    }
}

impl Add for PDF {
    type Output = PDF;
    fn add(self, rhs: PDF) -> Self::Output {
        PDF::from(self.0 + rhs.0)
    }
}
impl AddAssign for PDF {
    fn add_assign(&mut self, rhs: PDF) {
        self.0 += rhs.0;
    }
}

impl Mul<f32> for PDF {
    type Output = PDF;
    fn mul(self, rhs: f32) -> Self::Output {
        PDF::from(self.0 * rhs)
    }
}
impl Mul<PDF> for f32 {
    type Output = PDF;
    fn mul(self, rhs: PDF) -> Self::Output {
        PDF::from(self * rhs.0)
    }
}

impl Mul for PDF {
    type Output = PDF;
    fn mul(self, rhs: PDF) -> Self::Output {
        PDF::from(self.0 * rhs.0)
    }
}

impl MulAssign for PDF {
    fn mul_assign(&mut self, other: PDF) {
        self.0 = self.0 * other.0
    }
}
impl Div<f32> for PDF {
    type Output = PDF;
    fn div(self, rhs: f32) -> Self::Output {
        PDF::from(self.0 / rhs)
    }
}

// don't like copying this like this but i'm not sure how to do this using a macro so

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct PDFx4(pub f32x4);
impl PDFx4 {
    pub fn is_nan(&self) -> bool {
        self.0.is_nan().any()
    }
}

impl From<f32x4> for PDFx4 {
    fn from(val: f32x4) -> Self {
        PDFx4(val)
    }
}

impl From<PDFx4> for f32x4 {
    fn from(val: PDFx4) -> Self {
        val.0
    }
}

impl Add for PDFx4 {
    type Output = PDFx4;
    fn add(self, rhs: PDFx4) -> Self::Output {
        PDFx4::from(self.0 + rhs.0)
    }
}
impl AddAssign for PDFx4 {
    fn add_assign(&mut self, rhs: PDFx4) {
        self.0 += rhs.0;
    }
}

impl Mul<f32> for PDFx4 {
    type Output = PDFx4;
    fn mul(self, rhs: f32) -> Self::Output {
        PDFx4::from(self.0 * rhs)
    }
}
impl Mul<PDFx4> for f32 {
    type Output = PDFx4;
    fn mul(self, rhs: PDFx4) -> Self::Output {
        PDFx4::from(self * rhs.0)
    }
}

impl Mul for PDFx4 {
    type Output = PDFx4;
    fn mul(self, rhs: PDFx4) -> Self::Output {
        PDFx4::from(self.0 * rhs.0)
    }
}

impl MulAssign for PDFx4 {
    fn mul_assign(&mut self, other: PDFx4) {
        self.0 = self.0 * other.0
    }
}
impl Div<f32> for PDFx4 {
    type Output = PDFx4;
    fn div(self, rhs: f32) -> Self::Output {
        PDFx4::from(self.0 / rhs)
    }
}

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
