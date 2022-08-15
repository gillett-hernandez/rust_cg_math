#[macro_use]
extern crate packed_simd;

pub mod prelude;
pub mod traits;

pub mod bounds;
pub mod color;
pub mod curves;
pub mod misc;
pub mod pdf;
pub mod point;
pub mod random;
pub mod ray;
pub mod sample;
pub mod spectral;
pub mod tangent_frame;
pub mod transform;
pub mod vec;

use std::fmt::Debug;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Copy, Clone, PartialEq, Debug)]
pub enum Sidedness {
    Forward,
    Reverse,
    Dual,
}
