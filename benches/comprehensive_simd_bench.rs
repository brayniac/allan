use allan::*;
use std::time::Instant;

fn benchmark_with_config(name: &str, style: Style, max_tau: usize, samples: usize) {
    println!("\n=== {} ===", name);

    // Test Allan variance (benefits from batch updates)
    let mut allan = Allan::builder()
        .max_tau(max_tau)
        .style(style)
        .build_allan();

    let start = Instant::now();
    for i in 0..samples {
        allan.record((i as f64).sin() * 10.0 + (i as f64 / 7.0).cos() * 3.0);
    }
    let record_time = start.elapsed();

    // Test iteration (benefits from SIMD variance calculation)
    let start = Instant::now();
    let mut checksum = 0.0;
    for tau in allan.iter() {
        checksum += tau.variance().unwrap();
    }
    let iter_time = start.elapsed();

    #[cfg(not(feature = "simd"))]
    let mode = "Without SIMD";
    #[cfg(feature = "simd")]
    let mode = "With SIMD";

    println!("  {} - Record: {:?} ({:?}/sample), Iter: {:?}",
             mode, record_time, record_time / samples as u32, iter_time);
    println!("  Checksum: {:.6}", checksum);

    // Test Modified Allan (benefits from SIMD averaging + batch updates)
    let mut modified = ModifiedAllan::builder()
        .max_tau(max_tau)
        .style(style)
        .build_modified_allan();

    let start = Instant::now();
    for i in 0..samples {
        modified.record((i as f64).sin() * 10.0 + (i as f64 / 7.0).cos() * 3.0);
    }
    let mod_record_time = start.elapsed();

    let start = Instant::now();
    let mut mod_checksum = 0.0;
    for tau in modified.iter() {
        mod_checksum += tau.variance().unwrap();
    }
    let mod_iter_time = start.elapsed();

    println!("  Modified Allan {} - Record: {:?} ({:?}/sample), Iter: {:?}",
             mode, mod_record_time, mod_record_time / samples as u32, mod_iter_time);
    println!("  Modified Checksum: {:.6}", mod_checksum);
}

fn main() {
    println!("=== Comprehensive SIMD Benchmark ===");
    println!("Testing various configurations to show SIMD benefits\n");

    // Test with AllTau (many tau values - good for batch processing)
    benchmark_with_config(
        "AllTau with max_tau=100 (100 tau values)",
        Style::AllTau,
        100,
        10_000
    );

    // Test with AllTau and larger max_tau
    benchmark_with_config(
        "AllTau with max_tau=500 (500 tau values)",
        Style::AllTau,
        500,
        5_000
    );

    // Test with DecadeDeci (fewer tau values)
    benchmark_with_config(
        "DecadeDeci with max_tau=1000 (~27 tau values)",
        Style::DecadeDeci,
        1000,
        10_000
    );

    // Test cumulative mode with many updates
    println!("\n=== Cumulative Mode Test (50,000 samples) ===");
    let mut allan = Allan::builder()
        .max_tau(100)
        .style(Style::AllTau)
        .mode(Mode::Cumulative)
        .build_allan();

    let start = Instant::now();
    for i in 0..50_000 {
        allan.record((i as f64).sin());
    }
    let time = start.elapsed();

    #[cfg(not(feature = "simd"))]
    println!("  Without SIMD: {:?} total, {:?} per sample", time, time / 50_000);
    #[cfg(feature = "simd")]
    println!("  With SIMD: {:?} total, {:?} per sample", time, time / 50_000);

    // Verify correctness with a few samples
    let t1 = allan.get(1).unwrap();
    let t10 = allan.get(10).unwrap();
    let t100 = allan.get(100).unwrap();
    println!("  Variance at tau=1: {:.6}", t1.variance().unwrap());
    println!("  Variance at tau=10: {:.6}", t10.variance().unwrap());
    println!("  Variance at tau=100: {:.6}", t100.variance().unwrap());
}