use allan::*;
use std::time::Instant;

fn main() {
    println!("=== Sliding Mode SIMD Benchmark ===");
    println!("Testing sliding window performance with SIMD optimizations\n");

    // Test configuration
    let max_tau = 100;
    let initial_samples = 300; // Fill buffer
    let sliding_samples = 10_000; // Additional samples in sliding mode

    // Create Allan with sliding mode
    let mut allan = Allan::builder()
        .max_tau(max_tau)
        .style(Style::AllTau)
        .mode(Mode::Sliding)
        .build_allan();

    // Fill the buffer first
    for i in 0..initial_samples {
        allan.record((i as f64).sin() * 10.0 + (i as f64 / 7.0).cos() * 3.0);
    }

    // Now benchmark sliding mode updates
    let start = Instant::now();
    for i in initial_samples..(initial_samples + sliding_samples) {
        allan.record((i as f64).sin() * 10.0 + (i as f64 / 7.0).cos() * 3.0);
    }
    let sliding_time = start.elapsed();

    #[cfg(not(feature = "simd"))]
    let mode = "Without SIMD";
    #[cfg(feature = "simd")]
    let mode = "With SIMD";

    println!("Allan Sliding Mode ({}):", mode);
    println!("  {} samples in sliding mode: {:?}", sliding_samples, sliding_time);
    println!("  Per sample: {:?}\n", sliding_time / sliding_samples as u32);

    // Verify results
    let mut checksum = 0.0;
    for tau in allan.iter() {
        checksum += tau.variance().unwrap();
    }
    println!("  Checksum: {:.6}", checksum);

    // Test Modified Allan in sliding mode
    let mut modified = ModifiedAllan::builder()
        .max_tau(max_tau)
        .style(Style::AllTau)
        .mode(Mode::Sliding)
        .build_modified_allan();

    // Fill the buffer
    for i in 0..initial_samples {
        modified.record((i as f64).sin() * 10.0 + (i as f64 / 7.0).cos() * 3.0);
    }

    // Benchmark sliding mode
    let start = Instant::now();
    for i in initial_samples..(initial_samples + sliding_samples) {
        modified.record((i as f64).sin() * 10.0 + (i as f64 / 7.0).cos() * 3.0);
    }
    let mod_sliding_time = start.elapsed();

    println!("\nModified Allan Sliding Mode ({}):", mode);
    println!("  {} samples in sliding mode: {:?}", sliding_samples, mod_sliding_time);
    println!("  Per sample: {:?}", mod_sliding_time / sliding_samples as u32);

    // Verify results
    let mut mod_checksum = 0.0;
    for tau in modified.iter() {
        mod_checksum += tau.variance().unwrap();
    }
    println!("  Checksum: {:.6}", mod_checksum);

    // Test with larger max_tau for more intensive computation
    println!("\n=== Higher Load Test (max_tau=500) ===");

    let mut allan_500 = Allan::builder()
        .max_tau(500)
        .style(Style::AllTau)
        .mode(Mode::Sliding)
        .build_allan();

    // Fill buffer
    for i in 0..1500 {
        allan_500.record((i as f64).sin());
    }

    // Benchmark sliding
    let start = Instant::now();
    for i in 1500..6500 {
        allan_500.record((i as f64).sin());
    }
    let time_500 = start.elapsed();

    println!("Allan with max_tau=500 ({}):", mode);
    println!("  5000 samples in sliding mode: {:?}", time_500);
    println!("  Per sample: {:?}", time_500 / 5000);
}