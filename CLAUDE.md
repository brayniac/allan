# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

allan is a Rust library implementing Allan Variance and Deviation for stability analysis of oscillators, gyroscopes, and other time-series data. It provides streaming variance and deviation calculations with pre-allocated data structures for efficient processing.

## Build Commands

```bash
# Build the library
cargo build

# Build for release
cargo build --release

# Run tests
cargo test

# Format code
cargo fmt

# Run linter (requires nightly)
cargo +nightly clippy
```

## Architecture

### Core Components

The library consists of a single module (`src/lib.rs`) with three main structures:

1. **Allan** - The main data structure that manages the calculation
   - Maintains a rolling buffer of samples
   - Calculates overlapping Allan variance/deviation
   - Supports various tau spacing styles (decade, linear, custom)

2. **Tau** - Represents a time bucket for stability metrics
   - Stores accumulated values and counts
   - Provides variance and deviation calculations

3. **Config** - Builder pattern for Allan configuration
   - `max_tau`: Maximum tau value (determines buffer size)
   - `style`: Spacing between tau values (Decade, AllTau, etc.)

### Calculation Method

The library uses overlapping Allan variance calculation:
- Maintains a circular buffer of size `2 * max_tau + 1`
- For each new sample, calculates variance for all configured tau values
- Formula: `variance = (x[i+2τ] - 2x[i+τ] + x[i])² / (2τ²n)`

### Style Options

- `SingleTau(usize)` - Single specified tau value
- `AllTau` - All tau values from 1 to max_tau
- `Decade` - Powers of 10 (1, 10, 100, ...)
- `DecadeDeci` - Decade with decimal steps (1, 2, 3...9, 10, 20, 30...)
- `Decade124` - Decade with 1, 2, 4 steps
- `Decade1248` - Decade with 1, 2, 4, 8 steps
- `Decade125` - Decade with 1, 2, 5 steps

## Testing

The library includes tests for white noise and pink noise verification in the main module. Tests validate that Allan deviation values fall within expected ranges for known noise types.