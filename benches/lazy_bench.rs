use allan::*;
use std::time::Instant;

fn main() {
    // Test with many tau values
    let mut allan = Allan::builder()
        .max_tau(1000)
        .style(Style::AllTau) // 1000 tau values!
        .build_allan();

    // Benchmark record() performance
    println!("Recording 10,000 samples with 1000 tau values (no get() calls)...");
    let start = Instant::now();
    for i in 0..10_000 {
        allan.record((i as f64).sin());
    }
    let elapsed = start.elapsed();
    println!("Time: {:?}", elapsed);
    println!("Per sample: {:?}", elapsed / 10_000);

    // Now benchmark with occasional get() calls
    let mut allan2 = Allan::builder()
        .max_tau(1000)
        .style(Style::AllTau)
        .build_allan();

    println!("\nRecording 10,000 samples with get() every 1000 samples...");
    let start = Instant::now();
    for i in 0..10_000 {
        allan2.record((i as f64).sin());
        if i % 1000 == 999 {
            // Only compute variance when needed
            let _ = allan2.get(1);
            let _ = allan2.get(10);
            let _ = allan2.get(100);
        }
    }
    let elapsed = start.elapsed();
    println!("Time: {:?}", elapsed);
    println!("Per sample: {:?}", elapsed / 10_000);

    // Verify correctness
    println!("\nVerifying results match:");
    for tau in [1, 10, 100, 1000] {
        let t1 = allan.get(tau);
        let t2 = allan2.get(tau);
        if let (Some(r1), Some(r2)) = (t1, t2) {
            let diff = (r1.variance().unwrap() - r2.variance().unwrap()).abs();
            assert!(diff < 1e-10, "Results differ for tau={}: {} vs {}", tau, r1.variance().unwrap(), r2.variance().unwrap());
            println!("tau={}: variance={:.6}", tau, r1.variance().unwrap());
        }
    }
    println!("Results match!");
}