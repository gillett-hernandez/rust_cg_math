# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules

* always confirm with the user before running git commands that have side effects (branch, checkout, restore, reset, etc)

## Project Overview

`math` is a Rust math library for computer graphics / physically-based rendering. It provides SIMD-accelerated geometric primitives (via [`thermite`](https://github.com/raygon-renderer/thermite) — generic over backend), spectral color handling, measure-theoretic PDF types, sampling strategies, and curve/SPD evaluation. It is used as a dependency by a path tracer.

## Build Commands

This crate builds on **stable Rust** (edition 2024). It no longer uses any nightly features — `#![feature(portable_simd)]` and all `std::simd` references were removed once `HeroWavelength` migrated to thermite's `Vector<R>`. Note that `S::f32x4` etc. throughout the code are thermite's `Simd::f32x4` associated types, not `std::simd`.

```bash
cargo build              # build with default features (serde)
cargo test               # run all tests
cargo test <test_name>   # run a single test by name
cargo test -- --nocapture  # run tests with stdout visible
```

## SIMD Backend Selection

Every geometry / hero-wavelength type is generic over a thermite backend or register type. **The caller picks the backend**; this crate provides no default. Type parameters you'll see:

- `S: thermite::simd::Simd` — chooses a whole backend (e.g. `Scalar`, `X86V3`). Provides `S::f32x4`, `S::f32x16`, etc. Used by `Vec3<S>`, `Point3<S>`, `XYZColor<S>`, `Matrix4x4<S>`, `Transform3<S>`, `TangentFrame<S>`, `Ray<S>`, `random_*<S>()`, `uv_to_direction<S>()`.
- `R: thermite::register::FloatRegister<Element = f32>` — chooses one specific f32 register. Used by `HeroWavelength<R>` (so the caller can pick native-width `<S as FloatSimd<f32>>::fxN` or a fixed-width `S::f32x4` independently).

Tests across the crate use `thermite::backend::scalar::Scalar` as the chosen backend for portability and determinism. Switch to `thermite::backend::x86_v3` in a CI matrix entry if you want SIMD-path coverage.

`THERMITE_GUIDE.md` at the repo root is the authoritative thermite API reference (trait hierarchy, masked variants, dispatch).

## Architecture

### Core Type System

- **`Vec3<S>`** (`vec.rs`): 3D vector backed by `Vector<S::f32x4>`. Lane 3 is held at 0. `Mul<Vec3<S>> for Vec3<S>` is **dot product** (returns `f32`), not component-wise multiplication — uses `LinAlg3Vector::dot3`. `cross()` uses `cross3::<false>` (the simple formula — `DOP=true`'s accuracy correction breaks anticommutativity bit-exactness in tests).
- **`Point3<S>`** (`point.rs`): 3D point backed by `Vector<S::f32x4>`. Lane 3 held at 1.0. `Point3 - Point3 -> Vec3` (uses `zero4()` to clear w). `Point3 ± Vec3 -> Point3`. Cannot add two `Point3`s.
- **`Ray<S>`** (`ray.rs`): Origin + direction + time + tmax. `Ray::new`/`new_with_time`/etc are `const fn` — they just assign fields.
- **Constants are functions, not `const`s.** `Vec3::x_axis()`, `Vec3::y_axis()`, `Vec3::z_axis()`, `Point3::origin()`, `XYZColor::black()`, `Matrix4x4::identity()`. The one exception: `Vec3::<S>::ZERO` survives as a real `const` because it forwards to `<Vector<S::f32x4> as NumericVector>::ZERO`.

### Measure Theory / PDF System

The library encodes measure-theoretic concepts at the type level:

- **`Field` trait** (`traits.rs`): Abstraction for types that can be integrated over. Implemented for `f32` directly and **blanket-impl'd for every `Vector<R: FloatRegister>`** (so any thermite f32 vector type plugs into `WavelengthEnergy`, `PDF`, `Curve`, etc. without per-type wiring). The blanket impl bounds on `Vector<R>` (the type constructor), not `V: FloatVector`, to dodge a coherence conflict with the `f32` impl.
- **`Scalar` trait**: `Field + PartialOrd` (only `f32`).
- **`Measure` trait** (`traits.rs`): Defines a mathematical measure with an associated `Space` parameterization. Implementations: `Length`, `Area`, `Volume`, `SolidAngle<P>`, `ProjectedSolidAngle`, `Throughput`, `DiskAreaMeasure`.
- **`SpaceParameterization` trait** (`spaces.rs`): Defines the domain/space for a measure. `DirectionalSector` uses `[f32; 3]` for directions (not `Vec3<S>`) so the trait stays backend-agnostic.
- **`PDF<T, M>`** (`pdf.rs`): A probability density function value parameterized by field type `T` and measure `M`. Supports measure conversions (e.g., `convert_to_solid_angle`, `convert_to_projected_solid_angle`). `Deref<Target=T>`.

### Spectral / Color

- **`WavelengthEnergy<L, E>`** (`spectral.rs`): Pairs a wavelength with an energy value.
  - `SingleWavelength = WavelengthEnergy<f32, f32>`.
  - `HeroWavelength<R> = WavelengthEnergy<Vector<R>, Vector<R>>` — caller picks `R` (e.g. `<X86V3 as FloatSimd<f32>>::fxN` for native width, or `<X86V3 as Simd>::f32x4` for a fixed 4-lane bundle).
  - `new_from_range` uses `Vector::indexed()` to lay out N evenly-spaced wavelengths; works at any lane count.
- **`XYZColor<S>`** (`color/xyz.rs`): CIE XYZ color backed by `Vector<S::f32x4>`.
- **CIE observers** (`spectral.rs`): Two flavors per channel: scalar `x_bar`/`y_bar`/`z_bar` (`f32 -> f32`) and generic SIMD `x_bar_v`/`y_bar_v`/`z_bar_v` (`V -> V` where `V: FloatVectorWithBits<Element = f32> + TranscendentalMath`). Operating in angstroms internally (input nm * 10).

### Curves and SPDs

- **`Curve`** (`curves.rs`): Enum with `Const`, `Linear` (uniformly-spaced + interpolation), `Tabulated` (arbitrary `(x, y)` pairs), `Polynomial`, `Cauchy`, `Exponential` / `InverseExponential` (sums of asymmetric Gaussians), `Blackbody`, and `Machine` (composable Add/Mul stack).
- **`CurveWithCDF`**: `Curve` + precomputed CDF for importance sampling.
- **`SpectralPowerDistributionFunction<T>`**: implemented for `T = f32` (scalar) and for `T = Vector<R>` (generic over any thermite f32 float register with transcendental math). `Curve::Linear`/`Tabulated`/`Machine` fall back to per-lane scalar `v.map(|l| self.evaluate(l))` in the vector impl — correct but not maximally vectorized. A thermite `gather_or` path can replace `Linear`'s map fallback if profiling demands it.
- **`InterpolationMode`**: `Linear`, `Nearest`, `Cubic`.

### Sampling

- **`Sample1D`, `Sample2D`, `Sample3D`** (`sample.rs`): Sample types with values in `[0, 1)`.
- **`Sampler` trait**: `draw_1d`, `draw_2d`, `draw_3d`. Implementations: `RandomSampler`, `StratifiedSampler`.
- **Sampling functions** (`random.rs`): `random_on_unit_sphere<S>`, `random_in_unit_sphere<S>`, `random_in_unit_disk<S>`, `random_cosine_direction<S>`, `random_to_sphere<S>`. All return `Vec3<S>`.

### Transforms

- **`Matrix4x4<S>`** (`transform.rs`): 4x4 matrix stored as `[Vector<S::f32x4>; 4]` — one f32x4 per column (**column-major**). `as_array()`/`from_array()` use flat `m[col * 4 + row]` layout. Mat-vec and mat-mat mul and `transpose` dispatch to thermite's `LinAlg4Vector::mat4_vec3_product` / `mat4_vec4_product` / `mat4_product` / `mat4_transpose` with `COLUMN_MAJOR=true` (bounded `S::f32x4: LinAlg4Register`). `try_inverse` is a scalar cofactor/adjugate routine (runs once at construction, so not vectorized).
- **`Transform3<S>`**: Stores forward and reverse (inverse) matrices. No external linear-algebra dep — the affine constructors build their inverses analytically: `from_translation` negates the shift, `from_scale` reciprocates, `from_axis_angle` (Rodrigues' formula) transposes the orthonormal rotation. `new_from_matrix(Matrix4x4)` handles arbitrary matrices via `Matrix4x4::try_inverse` (returns `None` if singular). Also `from_stack`, `from_vector_stack`.
- **`TangentFrame<S>`** (`tangent_frame.rs`): Orthonormal basis (tangent, bitangent, normal) with `to_world`/`to_local`. `from_normal` builds a frame from a single normal vector.

### Utility

- **`Bounds1D`, `Bounds2D`** (`bounds.rs`): Axis-aligned intervals/rectangles. Half-open: `contains` is `[lower, upper)`.
- **`misc.rs`**: Scalar and SIMD Gaussian (`gaussian` f64 / `gaussianf32` / `gaussian_v<V>`), blackbody (`blackbody` / `blackbody_v<V>`), power heuristic for MIS (`power_heuristic` / `power_heuristic_v<V>`), UV-direction conversions (`uv_to_direction<S>` / `direction_to_uv<S>`).
- **`FromScalar`/`ToScalar`** (`traits.rs`): Custom conversion traits between `f32` and thermite vectors (orphan rules block `From`/`Into`).

## Feature Flags

- **`serde`** (default): Enables serialization derives on key types.
- **`deepsize`**: Enables `DeepSizeOf` derives for memory profiling.

## Testing

Tests use **proptest** for property-based testing throughout. Most modules have their own `#[cfg(test)]` blocks with property tests validating mathematical invariants (roundtrips, orthogonality, normalization, measure properties, etc.). Tests pin a `type TestS = thermite::backend::scalar::Scalar;` so they run portably without ISA-specific intrinsics.
