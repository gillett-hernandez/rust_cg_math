#![warn(rust_2018_idioms, rust_2021_compatibility)]

pub mod bounds;
pub mod color;
pub mod curves;
pub mod dual;
pub mod misc;
pub mod pdf;
pub mod point;
pub mod prelude;
pub mod quantity;
pub mod random;
pub mod ray;
pub mod sample;
pub mod spaces;
pub mod spectral;
pub mod tangent_frame;
pub mod traits;
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
pub(crate) fn assert_approx_eq(a: f32, b: f32, epsilon: f32) {
    assert!(
        (a - b).abs() < epsilon,
        "a was {}, b was {}, diff = {}",
        a,
        b,
        (a - b).abs()
    );
}

#[cfg(test)]
mod test {

    use super::prelude::*;
    #[test]
    fn test_variance_methods() {
        let mut samples = Vec::new();
        let mut sum = 0.0;
        let mut sum_of_squares = 0.0;

        let n = 10000;

        for _ in 0..n {
            let sample = debug_random();

            samples.push(sample);

            sum += sample;
            sum_of_squares += sample * sample;
        }

        let estimate = sum / n as f32;
        let variance_0 = samples
            .iter()
            .map(|sample| (*sample - estimate).powi(2))
            .sum::<f32>()
            / n as f32;
        let variance_1 = sum_of_squares / n as f32 - estimate * estimate;

        // estimate of uniform [0,1] mean should be near 0.5
        assert!((estimate - 0.5).abs() < 0.05, "estimate {} too far from 0.5", estimate);
        // both variance calculations should approximately agree
        assert!(
            (variance_0 - variance_1).abs() < 0.001,
            "variance methods disagree: {} vs {}",
            variance_0,
            variance_1
        );
    }
}
