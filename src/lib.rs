#![feature(portable_simd)]
#![warn(rust_2018_idioms, rust_2021_compatibility)]

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

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Sidedness {
    Forward,
    Reverse,
    Dual,
}

#[cfg(test)]
mod test {

    use super::prelude::*;
    #[test]
    fn test_variance_methods() {
        let mut samples = Vec::new();
        let mut sum = 0.0;
        let mut sum_of_squares = 0.0;

        let n = 100000;

        for _ in 0..n {
            let sample = debug_random();

            samples.push(sample);

            sum += sample;
            sum_of_squares += sample * sample;
        }

        let true_integral = 0.5;
        let estimate = sum / n as f32;
        let variance_0 = samples
            .iter()
            .map(|sample| (*sample - estimate).powi(2))
            .sum::<f32>()
            / n as f32;
        let variance_1 = sum_of_squares / n as f32 - estimate * estimate;
        println!("{} {}", estimate, true_integral);
        println!("{:?} {:?}", variance_0, variance_1);
    }

    #[test]
    fn test_adaptive_stratified_sampling() {
        let complex_curve = Curve::Linear {
            signal: vec![],
            bounds: Bounds1D::new(0.0, 1.0),
            mode: InterpolationMode::Cubic,
        };
        // TODO: finish writing this
    }
}
