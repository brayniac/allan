# allan - variance and deviation tools for stability analysis

[![crates.io](https://img.shields.io/crates/v/allan.svg)](https://crates.io/crates/allan)
[![License](https://img.shields.io/crates/l/allan.svg)](#license)

A Rust implementation of Allan variance and deviation calculations for analyzing the stability and noise characteristics of time-series data. This is particularly useful for characterizing frequency standards, oscillators, gyroscopes, and other precision measurement instruments.

## Overview

Allan variance is a method of representing frequency stability in oscillators and other time-series data. Unlike standard deviation, Allan variance converges for most types of noise commonly found in physical systems and can distinguish between different noise types.

This library provides:
- Overlapping Allan variance (AVAR) and Allan deviation (ADEV) calculations
- Configurable tau (averaging time) ranges with multiple spacing options
- Streaming calculation with efficient circular buffer implementation
- Support for real-time analysis of continuous data streams

## Example

```rust
use allan::Allan;

// Create a new Allan variance calculator
let mut allan = Allan::new();

// Add data points from your measurements
for sample in measurements {
    allan.record(sample);
}

// Get the Allan deviation at τ=1
let tau_1 = allan.get(1).unwrap();
println!("Allan deviation at τ=1: {}", tau_1.deviation().unwrap());
println!("Allan variance at τ=1: {}", tau_1.variance().unwrap());
```

## Documentation

API documentation is available at [docs.rs/allan](https://docs.rs/allan/).

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.