//! Criterion micro-benchmarks for the hot geometric / spectral primitives.
//!
//! Run with `cargo bench`. The geometry and SIMD-curve benches are generic over
//! the thermite backend and are instantiated for several widths:
//!   * `scalar`  — portable 1-lane fallback (no intrinsics)
//!   * `x86_v2`  — SSE4.2 (128-bit)
//!   * `x86_v3`  — AVX2 + FMA (256-bit)
//!
//! The x86 backends invoke target-feature-gated intrinsics, so build for a CPU
//! that actually has them, e.g. `RUSTFLAGS="-C target-cpu=native" cargo bench`.

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use math::curves::Op;
use math::prelude::*;
use thermite::backend::scalar::Scalar;
use thermite::backend::x86_v2::X86V2;
use thermite::backend::x86_v3::X86V3;
use thermite::math::TranscendentalMath;
use thermite::register::{FloatRegister, LinAlg3Register, LinAlg4Register};
use thermite::simd::Simd;

/// Geometry + SIMD-curve benchmarks, generic over the thermite backend `S`.
/// All ops for one backend land in a single criterion group named `backend`.
fn bench_geometry<S>(c: &mut Criterion, backend: &str)
where
    S: Simd,
    S::f32x4: LinAlg3Register + LinAlg4Register + FloatRegister<Element = f32>,
    Vector<S::f32x4>: FloatVectorWithBits<Element = f32> + TranscendentalMath,
{
    let mut group = c.benchmark_group(backend);

    // --- Vec3 ---
    let a = Vec3::<S>::new(1.0, 2.0, 3.0);
    let b = Vec3::<S>::new(-4.0, 5.0, -6.0);
    group.bench_function("vec3_cross", |bn| {
        bn.iter(|| black_box(a).cross(black_box(b)))
    });
    group.bench_function("vec3_dot", |bn| bn.iter(|| black_box(a) * black_box(b)));
    group.bench_function("vec3_add", |bn| bn.iter(|| black_box(a) + black_box(b)));
    group.bench_function("vec3_normalized", |bn| {
        bn.iter(|| black_box(a).normalized())
    });

    // --- Matrix4x4 ---
    #[rustfmt::skip]
    let m = Matrix4x4::<S>::from_array([
        2.0, 0.5, 0.0, 0.0,
        1.0, 3.0, 0.0, 0.0,
        0.0, 0.7, 4.0, 0.0,
        1.0, 2.0, 3.0, 1.0,
    ]);
    let n = Matrix4x4::<S>::from_array([
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 5.0, 6.0, 7.0, 1.0,
    ]);
    let p = Point3::<S>::new(1.0, 2.0, 3.0);
    group.bench_function("mat4_mul_mat4", |bn| {
        bn.iter(|| black_box(m) * black_box(n))
    });
    group.bench_function("mat4_mul_vec3", |bn| {
        bn.iter(|| black_box(m) * black_box(a))
    });
    group.bench_function("mat4_mul_point3", |bn| {
        bn.iter(|| black_box(m) * black_box(p))
    });
    group.bench_function("mat4_transpose", |bn| {
        bn.iter(|| black_box(m).transpose())
    });
    group.bench_function("mat4_try_inverse", |bn| {
        bn.iter(|| black_box(m).try_inverse())
    });

    // --- Transform3 ---
    let axis = Vec3::<S>::new(0.0, 1.0, 0.0);
    let rot = Transform3::<S>::from_axis_angle(axis, 0.7);
    let trans = Transform3::<S>::from_translation(Vec3::<S>::new(1.0, 2.0, 3.0));
    group.bench_function("transform_from_axis_angle", |bn| {
        bn.iter(|| Transform3::<S>::from_axis_angle(black_box(axis), black_box(0.7)))
    });
    group.bench_function("transform_to_world_point", |bn| {
        bn.iter(|| black_box(rot).to_world(black_box(p)))
    });
    group.bench_function("transform_to_local_point", |bn| {
        bn.iter(|| black_box(rot).to_local(black_box(p)))
    });
    group.bench_function("transform_compose", |bn| {
        bn.iter(|| black_box(rot) * black_box(trans))
    });

    // --- Sampling ---
    let s2 = Sample2D::new(0.3, 0.7);
    let s3 = Sample3D::new(0.3, 0.7, 0.1);
    group.bench_function("random_cosine_direction", |bn| {
        bn.iter(|| random_cosine_direction::<S>(black_box(s2)))
    });
    group.bench_function("random_on_unit_sphere", |bn| {
        bn.iter(|| random_on_unit_sphere::<S>(black_box(s2)))
    });
    group.bench_function("random_in_unit_sphere", |bn| {
        bn.iter(|| random_in_unit_sphere::<S>(black_box(s3)))
    });

    // --- Concentration-controllable lobe samplers (AD auto-pdf) ---
    // Each is benched twice: the value-only `f32` warp (hot path, no
    // derivatives) and the `_pdf` variant that runs the same warp through the
    // dual-number `SampleField` core, so the AD overhead of producing the
    // solid-angle pdf is directly visible side by side.
    let n_phong = 8.0_f32;
    let alpha_ggx = 0.3_f32;
    group.bench_function("power_cosine_direction", |bn| {
        bn.iter(|| power_cosine_direction::<S>(black_box(s2), black_box(n_phong)))
    });
    group.bench_function("power_cosine_direction_pdf", |bn| {
        bn.iter(|| power_cosine_direction_pdf::<S>(black_box(s2), black_box(n_phong)))
    });
    group.bench_function("ggx_direction", |bn| {
        bn.iter(|| ggx_direction::<S>(black_box(s2), black_box(alpha_ggx)))
    });
    group.bench_function("ggx_direction_pdf", |bn| {
        bn.iter(|| ggx_direction_pdf::<S>(black_box(s2), black_box(alpha_ggx)))
    });
    // Baseline AD overhead reference: the cosine warp value vs. auto-pdf paths.
    group.bench_function("random_cosine_direction_pdf", |bn| {
        bn.iter(|| random_cosine_direction_pdf::<S>(black_box(s2)))
    });

    // --- SIMD curve evaluation (one native-width register of wavelengths) ---
    // The transcendental-heavy variants (Exponential / Blackbody) best show the
    // benefit of wider lanes; Polynomial / Cauchy are pure arithmetic.
    let lambda = Vector::<S::f32x4>::splat(550.0);
    let exponential = Curve::y_bar();
    let blackbody = Curve::Blackbody { temperature: 5500.0, boost: 1.0 };
    let polynomial = Curve::Polynomial {
        domain_range_mapping: [600.0, 200.0, 0.0, 1.0],
        coefficients: [0.5, 0.1, -0.2, 0.05, 0.01, 0.0, 0.0, 0.0],
    };
    let cauchy = Curve::Cauchy { a: 1.5, b: 5000.0 };
    group.bench_function("curve_simd_exponential", |bn| {
        bn.iter(|| black_box(&exponential).evaluate_power(black_box(lambda)))
    });
    group.bench_function("curve_simd_blackbody", |bn| {
        bn.iter(|| black_box(&blackbody).evaluate_power(black_box(lambda)))
    });
    group.bench_function("curve_simd_polynomial", |bn| {
        bn.iter(|| black_box(&polynomial).evaluate_power(black_box(lambda)))
    });
    group.bench_function("curve_simd_cauchy", |bn| {
        bn.iter(|| black_box(&cauchy).evaluate_power(black_box(lambda)))
    });

    // --- Ellipse SDF ---
    // The scalar `sd_ellipse` is backend-agnostic (runs identically in every
    // group) and serves as the per-lane baseline for the vectorized form, which
    // evaluates one query point per lane against the same ellipse — so the
    // speedup over the scalar baseline grows with the backend's lane width.
    let ellipse = (2.0_f32, 1.0_f32);
    let pt = (1.3_f32, 0.6_f32);
    let px = Vector::<S::f32x4>::splat(pt.0);
    let py = Vector::<S::f32x4>::splat(pt.1);
    group.bench_function("sd_ellipse", |bn| {
        bn.iter(|| sd_ellipse(black_box(pt), black_box(ellipse)))
    });
    group.bench_function("sd_ellipse_v", |bn| {
        bn.iter(|| sd_ellipse_v(black_box(px), black_box(py), black_box(ellipse)))
    });

    group.finish();
}

/// Scalar (`f32 -> f32`) evaluation of every `Curve` variant. Backend-agnostic,
/// so this runs once rather than per-backend.
fn bench_curve_variants(c: &mut Criterion) {
    let mut group = c.benchmark_group("curve_scalar");

    let bounds = Bounds1D::new(380.0, 780.0);
    let linear = Curve::from_function(
        |x| (x - 550.0).abs() / 100.0,
        64,
        bounds,
        InterpolationMode::Linear,
    );
    let cubic = Curve::from_function(
        |x| (x - 550.0).abs() / 100.0,
        64,
        bounds,
        InterpolationMode::Cubic,
    );
    let tabulated = Curve::Tabulated {
        signal: vec![
            (400.0, 0.1),
            (450.0, 0.3),
            (500.0, 0.6),
            (550.0, 0.9),
            (600.0, 0.7),
            (650.0, 0.4),
            (700.0, 0.2),
        ],
        mode: InterpolationMode::Linear,
    };
    let machine = Curve::Machine {
        seed: 1.0,
        list: vec![
            (Op::Mul, Curve::Cauchy { a: 1.5, b: 5000.0 }),
            (Op::Add, Curve::Const(0.25)),
        ],
    };

    // One representative input in the visible range for every variant.
    let cases: [(&str, Curve); 10] = [
        ("const", Curve::Const(0.5)),
        ("linear", linear),
        ("cubic", cubic),
        ("tabulated", tabulated),
        (
            "polynomial",
            Curve::Polynomial {
                domain_range_mapping: [600.0, 200.0, 0.0, 1.0],
                coefficients: [0.5, 0.1, -0.2, 0.05, 0.01, 0.0, 0.0, 0.0],
            },
        ),
        ("cauchy", Curve::Cauchy { a: 1.5, b: 5000.0 }),
        ("exponential", Curve::y_bar()),
        (
            "inverse_exponential",
            Curve::InverseExponential {
                signal: vec![(550.0, 30.0, 30.0, 0.8)],
            },
        ),
        (
            "blackbody",
            Curve::Blackbody { temperature: 5500.0, boost: 1.0 },
        ),
        ("machine", machine),
    ];

    for (name, curve) in &cases {
        group.bench_function(*name, |bn| {
            bn.iter(|| black_box(curve).evaluate(black_box(550.0)))
        });
    }
    group.finish();
}

fn bench_scalar(c: &mut Criterion) {
    bench_geometry::<Scalar>(c, "scalar");
}
fn bench_x86_v2(c: &mut Criterion) {
    bench_geometry::<X86V2>(c, "x86_v2");
}
fn bench_x86_v3(c: &mut Criterion) {
    bench_geometry::<X86V3>(c, "x86_v3");
}

criterion_group!(
    benches,
    bench_scalar,
    bench_x86_v2,
    bench_x86_v3,
    bench_curve_variants
);
criterion_main!(benches);
