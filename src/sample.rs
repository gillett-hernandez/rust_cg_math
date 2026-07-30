use crate::prelude::*;
use rand::rngs::Xoshiro256PlusPlus;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng, rng};

use std::f32::EPSILON;
// TODO: add measure generic like with pdf to define what measure a sample is obtained wrt

#[derive(Debug, Copy, Clone)]
pub struct Sample1D {
    pub x: f32,
}

impl Sample1D {
    pub fn new(x: f32) -> Self {
        debug_assert!(x < 1.0 && x >= 0.0);
        Sample1D { x }
    }
    pub fn new_random_sample() -> Self {
        Sample1D::new(debug_random())
    }
    pub fn choose<T>(mut self, split: f32, a: T, b: T) -> (Self, T) {
        debug_assert!(0.0 <= split && split <= 1.0);
        debug_assert!(self.x >= 0.0 && self.x < 1.0);
        if self.x < split {
            assert!(split > 0.0);
            self.x /= split;
            (self, a)
        } else {
            // if split was 1.0, there's no way for self.x to be greather than or equal to it
            // since self.x in [0, 1)
            debug_assert!(split < 1.0);
            self.x = (self.x - split) / (1.0 - split);
            (self, b)
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Sample2D {
    pub x: f32,
    pub y: f32,
}

impl Sample2D {
    pub fn new(x: f32, y: f32) -> Self {
        debug_assert!(x < 1.0 && x >= 0.0);
        debug_assert!(y < 1.0 && y >= 0.0);

        Sample2D { x, y }
    }
    pub fn new_random_sample() -> Self {
        Sample2D::new(debug_random(), debug_random())
    }
}
#[derive(Debug, Copy, Clone)]
pub struct Sample3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Sample3D {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Sample3D { x, y, z }
    }
    pub fn new_random_sample() -> Self {
        Sample3D::new(debug_random(), debug_random(), debug_random())
    }
}

pub trait Sampler {
    fn draw_1d(&mut self) -> Sample1D;
    fn draw_2d(&mut self) -> Sample2D;
    fn draw_3d(&mut self) -> Sample3D;
}

/// SplitMix64's finalizing mix. A bijection on u64 that avalanches every input bit, so
/// seeds differing in a single bit produce uncorrelated generator streams.
#[inline]
const fn splitmix64_mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Derive a stable seed for one unit of work from a render's base seed.
///
/// The point of routing every per-work-unit seed through here is that the result depends only
/// on `(base_seed, unit_index, sample_index)` — never on thread scheduling, work-stealing
/// order, or how many samplers were constructed before this one. That is what makes a
/// parallel render reproducible.
///
/// `unit_index` and `sample_index` are packed into disjoint 32-bit halves before mixing, so
/// within their documented range (each < 2^32 — any film and sample count that can actually be
/// rendered) **no two work units share a seed**. `splitmix64_mix` is a bijection, so the
/// packing's injectivity survives the mix.
#[inline]
pub const fn derive_seed(base_seed: u64, unit_index: u64, sample_index: u64) -> u64 {
    debug_assert!(unit_index < (1 << 32));
    debug_assert!(sample_index < (1 << 32));
    let key = (unit_index << 32) | (sample_index & 0xFFFF_FFFF);
    splitmix64_mix(splitmix64_mix(base_seed) ^ key)
}

/// Uncorrelated uniform draws on `[0, 1)`, backed by an **owned** generator.
///
/// The generator is `Xoshiro256PlusPlus`, one of `rand`'s *named portable* generators: its
/// algorithm carries a reproducibility guarantee across `rand` releases and platforms, unlike
/// `SmallRng`/`StdRng`. A seed recorded in a bug report therefore still reproduces after a
/// dependency bump — which is the entire reason seeding exists here.
pub struct RandomSampler {
    rng: Xoshiro256PlusPlus,
}

impl RandomSampler {
    /// Seeded from OS entropy. A render driven by this is **not** reproducible — prefer
    /// [`RandomSampler::from_seed`] anywhere the result needs to be repeatable.
    pub fn new() -> RandomSampler {
        RandomSampler {
            rng: Xoshiro256PlusPlus::from_rng(&mut rng()),
        }
    }

    /// Deterministic: the full draw sequence is a pure function of `seed`. Per-work-unit seeds
    /// should come from [`derive_seed`] rather than from arithmetic on a base seed, so that
    /// adjacent units don't march in lockstep.
    pub fn from_seed(seed: u64) -> RandomSampler {
        RandomSampler {
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }

    /// `rand`'s f32 sampling fills a 24-bit mantissa, giving exactly `[0, 1)` — the same
    /// distribution `debug_random()` produced, so seeding changes reproducibility only and
    /// never the statistics of a render.
    #[inline(always)]
    fn next_f32(&mut self) -> f32 {
        self.rng.random::<f32>()
    }
}

impl Default for RandomSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for RandomSampler {
    fn draw_1d(&mut self) -> Sample1D {
        Sample1D::new(self.next_f32())
    }
    fn draw_2d(&mut self) -> Sample2D {
        let (x, y) = (self.next_f32(), self.next_f32());
        Sample2D::new(x, y)
    }
    fn draw_3d(&mut self) -> Sample3D {
        let (x, y, z) = (self.next_f32(), self.next_f32(), self.next_f32());
        Sample3D::new(x, y, z)
    }
}

// TODO: update stratified sampler implementation to use less memory, or deprecate it and add a low discrepancy sequence instead. https://en.wikipedia.org/wiki/Low-discrepancy_sequence
pub struct StratifiedSampler {
    pub dims: [usize; 3],
    pub indices: [usize; 3],
    pub first: Vec<usize>,
    pub second: Vec<usize>,
    pub third: Vec<usize>,
    // Concrete rather than `Box<dyn Rng>`: this drives both the stratum shuffles *and* the
    // within-stratum jitter, so it must be the sampler's single entropy source for
    // `from_seed` to mean anything. Being concrete also drops a dyn dispatch per shuffle.
    rng: Xoshiro256PlusPlus,
}

impl StratifiedSampler {
    /// Seeded from OS entropy; see [`RandomSampler::new`].
    pub fn new(xdim: usize, ydim: usize, zdim: usize) -> Self {
        Self::with_rng(xdim, ydim, zdim, Xoshiro256PlusPlus::from_rng(&mut rng()))
    }

    /// Deterministic: both the shuffles and the jitter derive from `seed`.
    pub fn from_seed(xdim: usize, ydim: usize, zdim: usize, seed: u64) -> Self {
        Self::with_rng(
            xdim,
            ydim,
            zdim,
            Xoshiro256PlusPlus::seed_from_u64(seed),
        )
    }

    fn with_rng(xdim: usize, ydim: usize, zdim: usize, rng: Xoshiro256PlusPlus) -> Self {
        StratifiedSampler {
            dims: [xdim, ydim, zdim],
            indices: [0, 0, 0],
            first: (0..xdim).into_iter().collect(),
            second: (0..(xdim * ydim)).into_iter().collect(),
            third: (0..(xdim * ydim * zdim)).into_iter().collect(),
            rng,
        }
    }

    /// The within-stratum jitter. Must go through `self.rng` — pulling it from thread-local
    /// entropy would leave the sampler unreproducible while *looking* seeded.
    #[inline(always)]
    fn next_f32(&mut self) -> f32 {
        self.rng.random::<f32>()
    }
}

impl Sampler for StratifiedSampler {
    fn draw_1d(&mut self) -> Sample1D {
        if self.indices[0] == 0 {
            // shuffle, then draw.
            self.first.shuffle(&mut self.rng);
            // print!("#");
        }
        let idx = self.first[self.indices[0]];
        let (width, _depth, _height) = (self.dims[0], self.dims[1], self.dims[2]);
        self.indices[0] += 1;
        if self.indices[0] >= width {
            self.indices[0] = 0;
        }
        // convert idx to the "pixel" based on dims
        let mut sample = Sample1D::new(self.next_f32());
        let x = idx;
        let old_x = sample.x;
        sample.x = (sample.x + x as f32) / (width as f32);
        if sample.x == 1.0 {
            sample.x -= EPSILON;
        }
        debug_assert!(
            sample.x < 1.0 && sample.x >= 0.0,
            "{:?} = ({:?} + {:?})/{:?}",
            sample.x,
            old_x,
            x,
            width,
        );
        sample
    }
    fn draw_2d(&mut self) -> Sample2D {
        if self.indices[1] == 0 {
            // shuffle, then draw.
            self.second.shuffle(&mut self.rng);
            // print!("#");
        }
        let idx = self.second[self.indices[1]];
        let (width, depth, _height) = (self.dims[0], self.dims[1], self.dims[2]);
        self.indices[1] += 1;
        if self.indices[1] >= width * depth {
            self.indices[1] = 0;
        }
        // convert idx to the "pixel" based on dims
        let (x, y) = (idx % width, idx / width);
        let mut sample = {
            let (jx, jy) = (self.next_f32(), self.next_f32());
            Sample2D::new(jx, jy)
        };
        let old_x = sample.x;
        sample.x = (sample.x + x as f32) / (width as f32);
        let old_y = sample.y;
        sample.y = (sample.y + y as f32) / (depth as f32);
        if sample.x == 1.0 {
            sample.x -= EPSILON;
        }
        if sample.y == 1.0 {
            sample.y -= EPSILON;
        }
        debug_assert!(
            sample.x < 1.0 && sample.x >= 0.0,
            "{:?} = ({:?} + {:?})/{:?}",
            sample.x,
            old_x,
            x,
            width,
        );
        debug_assert!(
            sample.y < 1.0 && sample.y >= 0.0,
            "{:?} = ({:?} + {:?})/{:?}",
            sample.y,
            old_y,
            y,
            depth,
        );
        sample
    }
    fn draw_3d(&mut self) -> Sample3D {
        if self.indices[2] == 0 {
            // shuffle, then draw.
            self.third.shuffle(&mut self.rng);
            // print!("#");
        }
        let idx = self.third[self.indices[2]];
        let (width, depth, height) = (self.dims[0], self.dims[1], self.dims[2]);
        self.indices[2] += 1;
        if self.indices[2] >= width * depth * height {
            self.indices[2] = 0;
        }
        // idx = x + width * y + width * depth * z
        // convert idx to the "pixel" based on dims
        // z coordinate is how many slices high the sample is
        let z = idx / (depth * width);
        // y coordinate is how far into a slice a given "pixel" is
        let y = (idx / width) % depth;
        // x coordinate is how far along width a given pixel is
        let x = idx % width;
        let mut sample = {
            let (jx, jy, jz) = (self.next_f32(), self.next_f32(), self.next_f32());
            Sample3D::new(jx, jy, jz)
        };
        sample.x = (sample.x + x as f32) / (width as f32);
        sample.y = (sample.y + y as f32) / (depth as f32);
        sample.z = (sample.z + z as f32) / (height as f32);
        if sample.x == 1.0 {
            sample.x -= EPSILON;
        }

        if sample.y == 1.0 {
            sample.y -= EPSILON;
        }
        if sample.z == 1.0 {
            sample.z -= EPSILON;
        }
        debug_assert!(sample.x < 1.0 && sample.x >= 0.0);
        debug_assert!(sample.y < 1.0 && sample.y >= 0.0);
        debug_assert!(sample.z < 1.0 && sample.z >= 0.0);
        sample
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn sample1d_choose_returns_valid_sample(x in 0.0f32..0.9999, split in 0.01f32..0.99) {
            let s = Sample1D { x };
            let (new_s, _chosen) = s.choose(split, "a", "b");
            prop_assert!(new_s.x >= 0.0 && new_s.x < 1.0 + 1e-6, "choose produced x={}", new_s.x);
        }

        #[test]
        fn sample1d_choose_picks_correctly(x in 0.0f32..0.9999, split in 0.01f32..0.99) {
            let s = Sample1D { x };
            let (_, chosen) = s.choose(split, "a", "b");
            if x < split {
                prop_assert_eq!(chosen, "a");
            } else {
                prop_assert_eq!(chosen, "b");
            }
        }
    }

    fn function(x: f32) -> f32 {
        x * x - x + 1.0
    }
    // true integral of x^2 - x + 1 on [0,1] = 1/3 - 1/2 + 1 = 5/6
    const TRUE_INTEGRAL: f32 = 5.0 / 6.0;

    #[test]
    fn test_random_sampler_1d() {
        let mut sampler = Box::new(RandomSampler::new());
        let n = 100000;
        let mut s = 0.0;
        for _i in 0..n {
            let sample = sampler.draw_1d();
            assert!(0.0 <= sample.x && sample.x < 1.0, "{}", sample.x);
            s += function(sample.x);
        }
        let estimate = s / n as f32;
        assert!(
            (estimate - TRUE_INTEGRAL).abs() < 0.01,
            "estimate {} too far from true integral {}",
            estimate,
            TRUE_INTEGRAL
        );
    }
    #[test]
    fn test_random_sampler_2d_and_3d() {
        let mut sampler = RandomSampler::new();
        for _ in 0..10000 {
            let s2 = sampler.draw_2d();
            assert!(0.0 <= s2.x && s2.x < 1.0, "2d x={}", s2.x);
            assert!(0.0 <= s2.y && s2.y < 1.0, "2d y={}", s2.y);
            let s3 = sampler.draw_3d();
            assert!(0.0 <= s3.x && s3.x < 1.0, "3d x={}", s3.x);
            assert!(0.0 <= s3.y && s3.y < 1.0, "3d y={}", s3.y);
            assert!(0.0 <= s3.z && s3.z < 1.0, "3d z={}", s3.z);
        }
    }

    /// Collect a sampler's draws as raw bits, so the comparison is exact and NaN-safe.
    fn draw_bits(sampler: &mut impl Sampler, count: usize) -> Vec<u32> {
        let mut bits = Vec::with_capacity(count * 6);
        for _ in 0..count {
            bits.push(sampler.draw_1d().x.to_bits());
            let s2 = sampler.draw_2d();
            bits.extend_from_slice(&[s2.x.to_bits(), s2.y.to_bits()]);
            let s3 = sampler.draw_3d();
            bits.extend_from_slice(&[s3.x.to_bits(), s3.y.to_bits(), s3.z.to_bits()]);
        }
        bits
    }

    /// A chi-square-ish uniformity check over `BINS` equal bins. Loose on purpose — this is
    /// guarding against gross structure (all draws in one bin, lockstep streams), not testing
    /// Xoshiro's statistical quality, which is not our job.
    fn assert_roughly_uniform(values: &[f32], what: &str) {
        const BINS: usize = 16;
        let mut histogram = [0usize; BINS];
        for &v in values {
            assert!(
                (0.0..1.0).contains(&v),
                "{what}: draw {v} outside [0, 1)"
            );
            histogram[(v * BINS as f32) as usize] += 1;
        }
        let expected = values.len() as f64 / BINS as f64;
        // Poisson-ish: the count in a bin has sd ~sqrt(expected); 5 sd is a very wide band
        // that still excludes any structured failure by orders of magnitude.
        let tolerance = 5.0 * expected.sqrt();
        for (bin, &count) in histogram.iter().enumerate() {
            assert!(
                (count as f64 - expected).abs() < tolerance,
                "{what}: bin {bin} held {count}, expected {expected:.1} +/- {tolerance:.1} \
                 (histogram {histogram:?})"
            );
        }
    }

    #[test]
    fn random_sampler_from_seed_is_reproducible() {
        let left = draw_bits(&mut RandomSampler::from_seed(0xC0FFEE), 10_000);
        let right = draw_bits(&mut RandomSampler::from_seed(0xC0FFEE), 10_000);
        assert_eq!(
            left, right,
            "two samplers built from the same seed must produce identical draw sequences"
        );

        // Non-vacuity: a sampler that always returned the same constant would pass the above.
        let other = draw_bits(&mut RandomSampler::from_seed(0xC0FFEF), 10_000);
        assert_ne!(
            left, other,
            "different seeds must produce different draw sequences"
        );
    }

    #[test]
    fn stratified_sampler_from_seed_is_reproducible() {
        // Exercises both entropy sources: the stratum shuffles *and* the within-stratum jitter.
        // Before seeding, the jitter came from `Sample*::new_random_sample()` — a second,
        // thread-local source that would leave this red while the shuffles looked seeded.
        let left = draw_bits(&mut StratifiedSampler::from_seed(20, 20, 10, 7), 5_000);
        let right = draw_bits(&mut StratifiedSampler::from_seed(20, 20, 10, 7), 5_000);
        assert_eq!(left, right, "stratified sampler must be reproducible by seed");

        let other = draw_bits(&mut StratifiedSampler::from_seed(20, 20, 10, 8), 5_000);
        assert_ne!(left, other, "different seeds must give different sequences");
    }

    #[test]
    fn derive_seed_decorrelates_adjacent_units() {
        const UNITS: u64 = 4096;
        const BASE: u64 = 0x5EED;

        // Injectivity: within the documented range no two work units may share a seed. This is
        // exact, not statistical — the (unit, sample) packing guarantees it and `splitmix64` is
        // a bijection.
        let mut seeds = Vec::with_capacity(UNITS as usize * 4);
        for unit in 0..UNITS {
            for sample in 0..4 {
                seeds.push(derive_seed(BASE, unit, sample));
            }
        }
        let mut sorted = seeds.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            seeds.len(),
            "derive_seed collided: {} distinct seeds for {} work units",
            sorted.len(),
            seeds.len()
        );

        // Then the downstream property: neighbouring units' *first* draws must be independent.
        // `Xoshiro256PlusPlus::seed_from_u64` splitmixes internally, so this survives even a
        // naive `seed = base + unit`; the guard is aimed at a future swap to `from_seed(bytes)`
        // (no mixing), where lockstep streams would show up as a structured image-wide
        // artifact rather than a crash. Verified red against a `derive_seed` that drops
        // `unit_index`: 1 distinct seed for 16384 work units.
        let first_draws: Vec<f32> = (0..UNITS)
            .map(|unit| RandomSampler::from_seed(derive_seed(BASE, unit, 0)).draw_1d().x)
            .collect();
        assert_roughly_uniform(&first_draws, "first draw across adjacent work units");

        // Adjacent units must not be nearly-equal to each other either — a lockstep generator
        // gives a near-constant successive difference.
        let mean_abs_delta: f64 = first_draws
            .windows(2)
            .map(|w| (w[1] - w[0]).abs() as f64)
            .sum::<f64>()
            / (first_draws.len() - 1) as f64;
        // E|U - V| = 1/3 for independent uniforms; a lockstep scheme drives this toward 0.
        assert!(
            (mean_abs_delta - 1.0 / 3.0).abs() < 0.02,
            "mean |first_draw(unit+1) - first_draw(unit)| = {mean_abs_delta:.4}, expected \
             ~0.3333 for independent streams (near 0 means adjacent units are correlated)"
        );
    }

    #[test]
    fn test_stratified_sampler_1d() {
        let mut sampler = Box::new(StratifiedSampler::new(20, 20, 10));
        let n = 100000;
        let mut s = 0.0;
        for _i in 0..n {
            let sample = sampler.draw_1d();
            assert!(0.0 <= sample.x && sample.x < 1.0, "{}", sample.x);
            s += function(sample.x);
        }
        let estimate = s / n as f32;
        assert!(
            (estimate - TRUE_INTEGRAL).abs() < 0.01,
            "estimate {} too far from true integral {}",
            estimate,
            TRUE_INTEGRAL
        );
    }
    #[test]
    fn test_stratified_sampler_2d() {
        let mut sampler = Box::new(StratifiedSampler::new(20, 20, 10));
        for _ in 0..10000 {
            sampler.draw_1d();
        }
        for _i in 0..100000 {
            let sample = sampler.draw_2d();
            assert!(0.0 <= sample.x && sample.x < 1.0, "{}", sample.x);
            assert!(0.0 <= sample.y && sample.y < 1.0, "{}", sample.y);
        }
    }
    #[test]
    fn test_stratified_sampler_3d() {
        let mut sampler = Box::new(StratifiedSampler::new(20, 20, 10));

        for _i in 0..100000 {
            let sample = sampler.draw_3d();
            assert!(0.0 <= sample.x && sample.x < 1.0, "{}", sample.x);
            assert!(0.0 <= sample.y && sample.y < 1.0, "{}", sample.y);
            assert!(0.0 <= sample.z && sample.z < 1.0, "{}", sample.z);
        }
    }
}
