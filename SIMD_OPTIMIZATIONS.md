# SIMD Optimizations Summary

This document describes the SIMD optimizations implemented in the allan variance library.

## Performance Improvements

Based on benchmark results, the SIMD optimizations provide:
- **Modified Allan variance**: 2.5-2.8x speedup for large tau counts
- **Allan variance sliding mode**: 1.4x speedup
- **Modified Allan sliding mode**: 1.3x speedup

## Implemented Optimizations

### 1. Batch Processing of Tau Values
Process 4 tau values simultaneously using `f64x4` SIMD vectors:
- Growing mode: Batch updates when buffer is filling
- Cumulative mode: Batch accumulation of new values
- Sliding mode: Batch subtraction of old values and addition of new values

### 2. SIMD-Accelerated Averaging (Modified Allan)
For Modified Allan variance, which requires averaging over tau intervals:
- Direct slicing when no buffer wrapping occurs
- Process 4 samples at a time using `f64x4::new()` and vector addition
- `reduce_add()` for efficient horizontal sum

### 3. Pre-computed Divisor Factors
Divisor factors are computed once during initialization to avoid repeated calculations:
```rust
divisor_factors: Vec<f64>, // Pre-computed tau^2 factors
```

### 4. Data Layout Optimization
Restructured from Array of Structs (AoS) to Struct of Arrays (SoA):
```rust
// Hot data separated for better cache locality
sums: Vec<f64>,
counts: Vec<u64>,
tau_values: Vec<u32>, // Cold data
```

## Benchmark Results

### Modified Allan with max_tau=500 (500 tau values):
- Without SIMD: 174µs per sample
- With SIMD: 65µs per sample
- **Speedup: 2.7x**

### Allan Sliding Mode:
- Without SIMD: 571ns per sample
- With SIMD: 406ns per sample
- **Speedup: 1.4x**

### Modified Allan Sliding Mode:
- Without SIMD: 20.64µs per sample
- With SIMD: 15.49µs per sample
- **Speedup: 1.3x**

## Implementation Details

The SIMD optimizations use the `wide` crate, which provides stable Rust SIMD support:
- Works on stable Rust (no nightly features required)
- Cross-platform support (x86_64 SSE/AVX and ARM NEON)
- Can be disabled with `--no-default-features` for platforms without SIMD

## Building and Testing

```bash
# Build with SIMD (default)
cargo build --release

# Build without SIMD
cargo build --release --no-default-features

# Run SIMD benchmarks
cargo bench --bench comprehensive_simd_bench
cargo bench --bench sliding_bench

# Compare with non-SIMD
cargo bench --bench comprehensive_simd_bench --no-default-features
```