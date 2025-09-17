use allan::*;
use std::time::Instant;

fn main() {
    println!("=== Modified Allan Variance SIMD Benchmark ===");
    println!("Testing with large tau values where SIMD helps most\n");

    // Test Modified Allan which benefits most from SIMD averaging
    let iterations = 10_000;

    // Benchmark without SIMD (compile without --features simd)
    let mut modified = ModifiedAllan::builder()
        .max_tau(100)  // Large tau means more averaging work
        .style(Style::AllTau)
        .build_modified_allan();

    println!("Recording {} samples with Modified Allan (max_tau=100)...", iterations);
    let start = Instant::now();
    for i in 0..iterations {
        modified.record((i as f64).sin() * 10.0 + (i as f64).cos() * 5.0);
    }
    let elapsed = start.elapsed();

    #[cfg(not(feature = "simd"))]
    println!("Without SIMD: {:?} total, {:?} per sample", elapsed, elapsed / iterations);

    #[cfg(feature = "simd")]
    println!("With SIMD: {:?} total, {:?} per sample", elapsed, elapsed / iterations);

    // Test retrieving values (triggers lazy computation)
    println!("\nComputing variances for all tau values...");
    let start = Instant::now();
    let mut sum = 0.0;
    for tau in 1..=100 {
        if let Some(t) = modified.get(tau) {
            if let Some(var) = t.variance() {
                sum += var;
            }
        }
    }
    let elapsed = start.elapsed();
    println!("Variance computation time: {:?}", elapsed);
    println!("Checksum (to prevent optimization): {}", sum);

    // Compare regular Allan for reference
    println!("\n=== Regular Allan Variance (for comparison) ===");

    let mut allan = Allan::builder()
        .max_tau(100)
        .style(Style::AllTau)
        .build_allan();

    println!("Recording {} samples with regular Allan...", iterations);
    let start = Instant::now();
    for i in 0..iterations {
        allan.record((i as f64).sin() * 10.0 + (i as f64).cos() * 5.0);
    }
    let elapsed = start.elapsed();
    println!("Time: {:?} total, {:?} per sample", elapsed, elapsed / iterations);

    // Large tau test for Modified Allan
    println!("\n=== Large Tau Test (tau=1000) ===");

    let mut modified_large = ModifiedAllan::builder()
        .max_tau(1000)
        .style(Style::SingleTau(1000))
        .build_modified_allan();

    println!("Recording 5000 samples with tau=1000...");
    let start = Instant::now();
    for i in 0..5000 {
        modified_large.record((i as f64).sin());
    }
    let elapsed = start.elapsed();

    #[cfg(not(feature = "simd"))]
    println!("Without SIMD: {:?}", elapsed);

    #[cfg(feature = "simd")]
    println!("With SIMD: {:?}", elapsed);

    if let Some(t) = modified_large.get(1000) {
        if let Some(var) = t.variance() {
            println!("Variance at tau=1000: {:.6}", var);
        }
    }
}