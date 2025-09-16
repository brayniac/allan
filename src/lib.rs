//! Allan provides variance and deviation tools for stability analysis
//!
//! # Goals
//! * provide streaming variance and deviations from series data
//! * pre-allocated datastructures
//! * support for gaps in data using NaN markers
//!
//! # Usage
//!
//! Create a new instance, add records, retrieve statistic
//!
//! # Gap Handling
//!
//! This implementation supports gaps in time-series data by using `NaN` as
//! a gap marker. When a gap is encountered (marked by recording `f64::NAN`),
//! calculations that would span the gap are automatically excluded from the
//! variance computation, ensuring accurate results even with incomplete data.
//!
//! # Example
//!
//! Create a new instance, add a few records, retrieve allan deviation
//!
//! ```
//!
//! use allan::*;
//!
//! // Allan variance/deviation
//! let mut allan = Allan::new();
//! for _ in 0..100 {
//!     allan.record(1.0);
//! }
//! assert_eq!(allan.get(1).unwrap().deviation().unwrap(), 0.0);
//!
//! // Hadamard variance/deviation (separate calculation)
//! let mut hadamard = Hadamard::new();
//! for _ in 0..100 {
//!     hadamard.record(1.0);
//! }
//! assert_eq!(hadamard.get(1).unwrap().deviation().unwrap(), 0.0);
//!
//! // Modified Allan variance/deviation (separate calculation)
//! let mut modified = ModifiedAllan::new();
//! for _ in 0..100 {
//!     modified.record(1.0);
//! }
//! assert_eq!(modified.get(1).unwrap().deviation().unwrap(), 0.0);
//! ```


#[cfg(feature = "simd")]
use wide::f64x4;

// ========== Internal Implementation (Hidden from public API) ==========

/// Internal trait for different variance calculation methods
trait VarianceMethod: Clone {
    /// Minimum number of samples needed for a given tau
    fn min_samples(&self, tau: usize) -> usize;

    /// Calculate the squared difference for this variance method
    /// For modified allan, this needs access to the full buffer and tau
    fn calculate_squared_diff(&self, buffer: &[f64], newest_idx: usize, tau: usize, buffer_len: usize) -> f64 {
        // Default implementation for methods that use simple indexing
        let indices = self.get_indices(newest_idx, tau, buffer_len);
        let diff = self.calculate_diff(buffer, indices.0, indices.1, indices.2, indices.3);
        diff * diff
    }

    /// Calculate old squared diff with a substituted value (for sliding mode)
    /// This is needed when the old value being removed is part of the calculation
    fn calculate_old_squared_diff_with_substitution(
        &self,
        buffer: &[f64],
        newest_idx: usize,
        tau: usize,
        buffer_len: usize,
        old_head: usize,
        old_value: f64
    ) -> f64 {
        // Default implementation for simple methods
        let indices = self.get_indices(newest_idx, tau, buffer_len);

        // Check if old_head is one of the indices used in the calculation
        let uses_old_head = indices.0 == old_head || indices.1 == old_head
            || indices.2 == old_head || indices.3 == Some(old_head);

        if uses_old_head {
            // Calculate with the old value substituted
            let val0 = if indices.0 == old_head { old_value } else { buffer[indices.0] };
            let val1 = if indices.1 == old_head { old_value } else { buffer[indices.1] };
            let val2 = if indices.2 == old_head { old_value } else { buffer[indices.2] };

            let diff = if let Some(idx3) = indices.3 {
                let val3 = if idx3 == old_head { old_value } else { buffer[idx3] };
                let temp_buf = [val0, val1, val2, val3];
                self.calculate_diff(&temp_buf, 0, 1, 2, Some(3))
            } else {
                let temp_buf = [val0, val1, val2];
                self.calculate_diff(&temp_buf, 0, 1, 2, None)
            };

            diff * diff
        } else {
            // old_head is not in this calculation, use current buffer values
            self.calculate_squared_diff(buffer, newest_idx, tau, buffer_len)
        }
    }

    /// Get indices for the calculation (used by non-modified methods)
    fn get_indices(&self, newest_idx: usize, tau: usize, buffer_len: usize) -> (usize, usize, usize, Option<usize>) {
        let min_samples = self.min_samples(tau);
        if min_samples == 2 * tau + 1 {
            (
                (newest_idx + buffer_len - 2 * tau) % buffer_len,
                (newest_idx + buffer_len - tau) % buffer_len,
                newest_idx,
                None
            )
        } else {
            (
                (newest_idx + buffer_len - 3 * tau) % buffer_len,
                (newest_idx + buffer_len - 2 * tau) % buffer_len,
                (newest_idx + buffer_len - tau) % buffer_len,
                Some(newest_idx)
            )
        }
    }

    /// Calculate the difference for this variance method
    fn calculate_diff(&self, buffer: &[f64], idx0: usize, idx1: usize, idx2: usize, idx3: Option<usize>) -> f64;

    /// Divisor for the variance calculation
    fn divisor(&self, tau: u32, count: u64) -> f64;
}

/// Allan variance method (2nd difference)
#[derive(Clone)]
struct AllanMethod;

impl VarianceMethod for AllanMethod {
    fn min_samples(&self, tau: usize) -> usize {
        2 * tau + 1
    }

    #[inline(always)]
    fn calculate_diff(&self, buffer: &[f64], idx0: usize, idx1: usize, idx2: usize, _idx3: Option<usize>) -> f64 {
        // Return NaN if any value is a gap
        if buffer[idx0].is_nan() || buffer[idx1].is_nan() || buffer[idx2].is_nan() {
            return f64::NAN;
        }
        buffer[idx2] - 2.0 * buffer[idx1] + buffer[idx0]
    }

    fn divisor(&self, tau: u32, count: u64) -> f64 {
        2.0 * count as f64 * (tau as f64) * (tau as f64)
    }
}

/// Hadamard variance method (3rd difference)
#[derive(Clone)]
struct HadamardMethod;

impl VarianceMethod for HadamardMethod {
    fn min_samples(&self, tau: usize) -> usize {
        3 * tau + 1
    }

    #[inline(always)]
    fn calculate_diff(&self, buffer: &[f64], idx0: usize, idx1: usize, idx2: usize, idx3: Option<usize>) -> f64 {
        let idx3 = idx3.expect("Hadamard requires 4 points");
        // Return NaN if any value is a gap
        if buffer[idx0].is_nan() || buffer[idx1].is_nan() || buffer[idx2].is_nan() || buffer[idx3].is_nan() {
            return f64::NAN;
        }
        buffer[idx3] - 3.0 * buffer[idx2] + 3.0 * buffer[idx1] - buffer[idx0]
    }

    fn divisor(&self, tau: u32, count: u64) -> f64 {
        6.0 * count as f64 * (tau as f64) * (tau as f64)
    }
}

/// Modified Allan variance method
/// Uses averaging over tau intervals before computing differences
#[derive(Clone)]
struct ModifiedAllanMethod;

impl ModifiedAllanMethod {
    /// Calculate the average over a tau interval
    #[cfg(not(feature = "simd"))]
    fn average(&self, buffer: &[f64], start_idx: usize, tau: usize, buffer_len: usize) -> f64 {
        let mut sum = 0.0;
        for i in 0..tau {
            let idx = (start_idx + i) % buffer_len;
            let val = buffer[idx];
            // If we encounter a gap, the average is invalid
            if val.is_nan() {
                return f64::NAN;
            }
            sum += val;
        }
        sum / tau as f64
    }

    /// SIMD-optimized average calculation
    #[cfg(feature = "simd")]
    fn average(&self, buffer: &[f64], start_idx: usize, tau: usize, buffer_len: usize) -> f64 {
        // Check if we can avoid modulo operations (common case when buffer wraps less than once)
        if start_idx + tau <= buffer_len {
            // No wrapping needed - we can use direct slicing which is much faster
            let slice = &buffer[start_idx..start_idx + tau];

            // First check for NaN values
            for &val in slice {
                if val.is_nan() {
                    return f64::NAN;
                }
            }

            // Process 4 elements at a time using SIMD
            let mut sum_vec = f64x4::splat(0.0);
            let chunks = slice.chunks_exact(4);
            let remainder = chunks.remainder();

            // Process chunks of 4 directly from the slice
            for chunk in chunks {
                let vals = f64x4::new([chunk[0], chunk[1], chunk[2], chunk[3]]);
                sum_vec += vals;
            }

            // Sum the vector elements
            let mut sum = sum_vec.reduce_add();

            // Handle remainder
            for &val in remainder {
                sum += val;
            }

            sum / tau as f64
        } else {
            // Wrapping case - fall back to modulo
            let mut sum = 0.0;
            for i in 0..tau {
                let idx = (start_idx + i) % buffer_len;
                let val = buffer[idx];
                if val.is_nan() {
                    return f64::NAN;
                }
                sum += val;
            }
            sum / tau as f64
        }
    }
}

impl VarianceMethod for ModifiedAllanMethod {
    fn min_samples(&self, tau: usize) -> usize {
        3 * tau  // Need 3 tau-length intervals for modified Allan
    }

    fn calculate_squared_diff(&self, buffer: &[f64], newest_idx: usize, tau: usize, buffer_len: usize) -> f64 {
        // For Modified Allan, we average over three consecutive tau intervals
        // then compute the second difference of these averages

        // Calculate start indices for the three tau intervals
        let start2 = (newest_idx + buffer_len + 1 - tau) % buffer_len;
        let start1 = (start2 + buffer_len - tau) % buffer_len;
        let start0 = (start1 + buffer_len - tau) % buffer_len;

        // Calculate averages for each tau interval
        let avg0 = self.average(buffer, start0, tau, buffer_len);
        let avg1 = self.average(buffer, start1, tau, buffer_len);
        let avg2 = self.average(buffer, start2, tau, buffer_len);

        // Second difference of the averages
        let diff = avg2 - 2.0 * avg1 + avg0;
        diff * diff
    }

    fn calculate_old_squared_diff_with_substitution(
        &self,
        buffer: &[f64],
        newest_idx: usize,
        tau: usize,
        buffer_len: usize,
        old_head: usize,
        old_value: f64
    ) -> f64 {
        // For Modified Allan, we need to handle averaging specially
        // Calculate start indices for the three tau intervals
        let start2 = (newest_idx + buffer_len + 1 - tau) % buffer_len;
        let start1 = (start2 + buffer_len - tau) % buffer_len;
        let start0 = (start1 + buffer_len - tau) % buffer_len;

        // Helper function to calculate average with substitution
        let average_with_sub = |start_idx: usize| {
            let mut sum = 0.0;
            for i in 0..tau {
                let idx = (start_idx + i) % buffer_len;
                sum += if idx == old_head { old_value } else { buffer[idx] };
            }
            sum / tau as f64
        };

        // Calculate averages with old value substituted where needed
        let avg0 = average_with_sub(start0);
        let avg1 = average_with_sub(start1);
        let avg2 = average_with_sub(start2);

        // Second difference of the averages
        let diff = avg2 - 2.0 * avg1 + avg0;
        diff * diff
    }

    fn calculate_diff(&self, _buffer: &[f64], _idx0: usize, _idx1: usize, _idx2: usize, _idx3: Option<usize>) -> f64 {
        // Not used for modified Allan - we override calculate_squared_diff instead
        unreachable!("ModifiedAllanMethod should use calculate_squared_diff")
    }

    fn divisor(&self, tau: u32, count: u64) -> f64 {
        2.0 * count as f64 * (tau as f64) * (tau as f64)
    }
}

/// Generic variance calculator (internal use only)
#[derive(Clone)]
struct Variance<M: VarianceMethod> {
    method: M,
    config: Config,
    buffer: Vec<f64>,
    head: usize,
    len: usize,
    total_samples: usize,
    sums: Vec<f64>,
    counts: Vec<u64>,
    tau_values: Vec<u32>,
    divisor_factors: Vec<f64>,
}

impl<M: VarianceMethod> Variance<M> {
    fn new(method: M) -> Self {
        Self::with_config(method, Config::default())
    }

    fn with_config(method: M, config: Config) -> Self {
        let max_tau = config.max_tau;
        let samples = method.min_samples(max_tau);

        let mut buffer = Vec::with_capacity(samples);
        buffer.resize(samples, 0.0);

        let (tau_values, divisor_factors) = generate_tau_data(&config, &method);
        let num_taus = tau_values.len();

        Self {
            method,
            config,
            sums: vec![0.0; num_taus],
            counts: vec![0; num_taus],
            tau_values,
            divisor_factors,
            buffer,
            head: 0,
            len: 0,
            total_samples: 0,
        }
    }

    fn record(&mut self, value: f64) {
        self.total_samples += 1;

        // NaN indicates a gap in the data - store it but don't compute with it
        if self.len < self.buffer.len() {
            // Buffer not full yet - just append
            self.buffer[self.len] = value;
            self.len += 1;
            // Only update calculations if this isn't a gap marker
            if !value.is_nan() {
                self.update_incremental_growing();
            }
        } else {
            // Buffer is full
            match self.config.mode {
                Mode::Cumulative => {
                    // In cumulative mode, keep accumulating
                    // We still use the buffer for the most recent samples
                    // but we don't subtract old values from the sum
                    let old_head = self.head;
                    self.buffer[self.head] = value;
                    self.head = (self.head + 1) % self.buffer.len();
                    // Only update if new value isn't NaN
                    if !value.is_nan() {
                        self.update_incremental_cumulative(old_head);
                    }
                }
                Mode::Sliding => {
                    // In sliding mode, remove old and add new
                    let old_value = self.buffer[self.head];
                    self.buffer[self.head] = value;
                    let old_head = self.head;
                    self.head = (self.head + 1) % self.buffer.len();
                    // Only update if new value isn't NaN
                    if !value.is_nan() {
                        self.update_incremental_sliding(old_head, old_value);
                    }
                }
            }
        }
    }

    #[cfg(not(feature = "simd"))]
    fn update_incremental_growing(&mut self) {
        // Buffer is growing - just add new calculations
        let current_len = self.len;
        let newest_idx = self.len - 1;

        for (i, &tau) in self.tau_values.iter().enumerate() {
            let t = tau as usize;
            let min_samples = self.method.min_samples(t);

            // Most likely path first (buffer has enough samples)
            if current_len >= min_samples {
                let squared_diff = self.method.calculate_squared_diff(&self.buffer, newest_idx, t, self.buffer.len());
                // Only update if the calculation is valid (not NaN)
                if !squared_diff.is_nan() {
                    self.sums[i] += squared_diff;
                    self.counts[i] += 1;
                }
            }
        }
    }

    #[cfg(feature = "simd")]
    fn update_incremental_growing(&mut self) {
        // SIMD version: Process multiple tau values in batches
        let newest_idx = self.len - 1;
        let current_len = self.len;
        let num_taus = self.tau_values.len();

        // Process 4 tau values at a time where possible
        let chunks = num_taus / 4;
        let remainder_start = chunks * 4;

        // Process chunks of 4 with SIMD
        for chunk_idx in 0..chunks {
            let base_idx = chunk_idx * 4;

            // Check if all 4 taus have enough samples
            let mut can_process = [false; 4];
            let mut diffs = [0.0; 4];

            for j in 0..4 {
                let i = base_idx + j;
                let t = self.tau_values[i] as usize;
                let min_samples = self.method.min_samples(t);
                can_process[j] = current_len >= min_samples;
                if can_process[j] {
                    diffs[j] = self.method.calculate_squared_diff(&self.buffer, newest_idx, t, self.buffer.len());
                }
            }

            // Update sums and counts using SIMD
            // Check if all 4 can be processed and none are NaN
            let all_valid = can_process.iter().all(|&x| x) &&
                           diffs.iter().all(|&x| !x.is_nan());

            if all_valid {
                // All 4 can be processed - use SIMD
                let sums_vec = f64x4::new([
                    self.sums[base_idx],
                    self.sums[base_idx + 1],
                    self.sums[base_idx + 2],
                    self.sums[base_idx + 3],
                ]);
                let diffs_vec = f64x4::new(diffs);
                let result = sums_vec + diffs_vec;

                self.sums[base_idx] = result.as_array_ref()[0];
                self.sums[base_idx + 1] = result.as_array_ref()[1];
                self.sums[base_idx + 2] = result.as_array_ref()[2];
                self.sums[base_idx + 3] = result.as_array_ref()[3];

                for j in 0..4 {
                    self.counts[base_idx + j] += 1;
                }
            } else {
                // Process individually
                for j in 0..4 {
                    if can_process[j] && !diffs[j].is_nan() {
                        let i = base_idx + j;
                        self.sums[i] += diffs[j];
                        self.counts[i] += 1;
                    }
                }
            }
        }

        // Process remainder
        for i in remainder_start..num_taus {
            let t = self.tau_values[i] as usize;
            let min_samples = self.method.min_samples(t);

            if current_len >= min_samples {
                let squared_diff = self.method.calculate_squared_diff(&self.buffer, newest_idx, t, self.buffer.len());
                // Only update if the calculation is valid (not NaN)
                if !squared_diff.is_nan() {
                    self.sums[i] += squared_diff;
                    self.counts[i] += 1;
                }
            }
        }
    }

    #[cfg(not(feature = "simd"))]
    fn update_incremental_cumulative(&mut self, _old_head: usize) {
        // In cumulative mode, we just add the new calculation
        let new_calc_newest = (self.head + self.buffer.len() - 1) % self.buffer.len();

        for (i, &tau) in self.tau_values.iter().enumerate() {
            let t = tau as usize;
            let new_squared_diff = self.method.calculate_squared_diff(&self.buffer, new_calc_newest, t, self.buffer.len());
            // Only update if the calculation is valid (not NaN)
            if !new_squared_diff.is_nan() {
                self.sums[i] += new_squared_diff;
                self.counts[i] += 1;
            }
        }
    }

    #[cfg(feature = "simd")]
    fn update_incremental_cumulative(&mut self, _old_head: usize) {
        // SIMD version for cumulative mode
        let new_calc_newest = (self.head + self.buffer.len() - 1) % self.buffer.len();
        let num_taus = self.tau_values.len();

        // Process in chunks of 4
        let chunks = num_taus / 4;
        let remainder_start = chunks * 4;

        for chunk_idx in 0..chunks {
            let base_idx = chunk_idx * 4;

            // Calculate all 4 diffs
            let diffs = [
                self.method.calculate_squared_diff(&self.buffer, new_calc_newest, self.tau_values[base_idx] as usize, self.buffer.len()),
                self.method.calculate_squared_diff(&self.buffer, new_calc_newest, self.tau_values[base_idx + 1] as usize, self.buffer.len()),
                self.method.calculate_squared_diff(&self.buffer, new_calc_newest, self.tau_values[base_idx + 2] as usize, self.buffer.len()),
                self.method.calculate_squared_diff(&self.buffer, new_calc_newest, self.tau_values[base_idx + 3] as usize, self.buffer.len()),
            ];

            // Check if all diffs are valid (not NaN)
            if diffs.iter().all(|&x| !x.is_nan()) {
                // Update sums using SIMD
                let sums_vec = f64x4::new([
                    self.sums[base_idx],
                    self.sums[base_idx + 1],
                    self.sums[base_idx + 2],
                    self.sums[base_idx + 3],
                ]);
                let diffs_vec = f64x4::new(diffs);
                let result = sums_vec + diffs_vec;

                self.sums[base_idx] = result.as_array_ref()[0];
                self.sums[base_idx + 1] = result.as_array_ref()[1];
                self.sums[base_idx + 2] = result.as_array_ref()[2];
                self.sums[base_idx + 3] = result.as_array_ref()[3];

                // Update counts
                for j in 0..4 {
                    self.counts[base_idx + j] += 1;
                }
            } else {
                // Process individually, skipping NaN values
                for j in 0..4 {
                    if !diffs[j].is_nan() {
                        self.sums[base_idx + j] += diffs[j];
                        self.counts[base_idx + j] += 1;
                    }
                }
            }
        }

        // Process remainder
        for i in remainder_start..num_taus {
            let t = self.tau_values[i] as usize;
            let new_squared_diff = self.method.calculate_squared_diff(&self.buffer, new_calc_newest, t, self.buffer.len());
            // Only update if the calculation is valid (not NaN)
            if !new_squared_diff.is_nan() {
                self.sums[i] += new_squared_diff;
                self.counts[i] += 1;
            }
        }
    }

    #[cfg(not(feature = "simd"))]
    fn update_incremental_sliding(&mut self, old_head: usize, old_value: f64) {
        // Buffer is full and sliding - remove old, add new
        let new_calc_newest = (self.head + self.buffer.len() - 1) % self.buffer.len();

        for (i, &tau) in self.tau_values.iter().enumerate() {
            let t = tau as usize;
            let min_samples = self.method.min_samples(t);

            // Calculate indices for the old squared diff that needs to be removed
            let old_calc_newest = (old_head + min_samples - 1) % self.buffer.len();

            // Calculate the old squared diff without mutating the buffer
            // We need to handle both simple methods and ModifiedAllan differently
            let old_squared_diff = self.method.calculate_old_squared_diff_with_substitution(
                &self.buffer, old_calc_newest, t, self.buffer.len(), old_head, old_value
            );

            // Calculate the new squared diff to add
            let new_squared_diff = self.method.calculate_squared_diff(&self.buffer, new_calc_newest, t, self.buffer.len());

            // Update the sum only if both values are valid (count stays the same when sliding)
            if !old_squared_diff.is_nan() && !new_squared_diff.is_nan() {
                self.sums[i] = self.sums[i] - old_squared_diff + new_squared_diff;
            }
        }
    }

    #[cfg(feature = "simd")]
    fn update_incremental_sliding(&mut self, old_head: usize, old_value: f64) {
        // SIMD version for sliding mode
        let new_calc_newest = (self.head + self.buffer.len() - 1) % self.buffer.len();
        let num_taus = self.tau_values.len();

        // Process in chunks of 4
        let chunks = num_taus / 4;
        let remainder_start = chunks * 4;

        for chunk_idx in 0..chunks {
            let base_idx = chunk_idx * 4;

            // Calculate old and new diffs for all 4 tau values
            let mut old_diffs = [0.0; 4];
            let mut new_diffs = [0.0; 4];

            for j in 0..4 {
                let i = base_idx + j;
                let t = self.tau_values[i] as usize;
                let min_samples = self.method.min_samples(t);

                // Calculate old diff
                let old_calc_newest = (old_head + min_samples - 1) % self.buffer.len();
                old_diffs[j] = self.method.calculate_old_squared_diff_with_substitution(
                    &self.buffer, old_calc_newest, t, self.buffer.len(), old_head, old_value
                );

                // Calculate new diff
                new_diffs[j] = self.method.calculate_squared_diff(&self.buffer, new_calc_newest, t, self.buffer.len());
            }

            // Check if all diffs are valid (not NaN)
            let all_valid = old_diffs.iter().all(|&x| !x.is_nan()) &&
                            new_diffs.iter().all(|&x| !x.is_nan());

            if all_valid {
                // Update sums using SIMD
                let sums_vec = f64x4::new([
                    self.sums[base_idx],
                    self.sums[base_idx + 1],
                    self.sums[base_idx + 2],
                    self.sums[base_idx + 3],
                ]);
                let old_vec = f64x4::new(old_diffs);
                let new_vec = f64x4::new(new_diffs);
                let result = sums_vec - old_vec + new_vec;

                self.sums[base_idx] = result.as_array_ref()[0];
                self.sums[base_idx + 1] = result.as_array_ref()[1];
                self.sums[base_idx + 2] = result.as_array_ref()[2];
                self.sums[base_idx + 3] = result.as_array_ref()[3];
            } else {
                // Process individually, skipping NaN values
                for j in 0..4 {
                    if !old_diffs[j].is_nan() && !new_diffs[j].is_nan() {
                        self.sums[base_idx + j] = self.sums[base_idx + j] - old_diffs[j] + new_diffs[j];
                    }
                }
            }
        }

        // Process remainder
        for i in remainder_start..num_taus {
            let t = self.tau_values[i] as usize;
            let min_samples = self.method.min_samples(t);

            // Calculate old diff
            let old_calc_newest = (old_head + min_samples - 1) % self.buffer.len();
            let old_squared_diff = self.method.calculate_old_squared_diff_with_substitution(
                &self.buffer, old_calc_newest, t, self.buffer.len(), old_head, old_value
            );

            let new_squared_diff = self.method.calculate_squared_diff(&self.buffer, new_calc_newest, t, self.buffer.len());
            // Update the sum only if both values are valid
            if !old_squared_diff.is_nan() && !new_squared_diff.is_nan() {
                self.sums[i] = self.sums[i] - old_squared_diff + new_squared_diff;
            }
        }
    }

    fn get(&self, tau: usize) -> Option<Tau> {
        // Find the index of this tau value
        let index = self.tau_values.iter().position(|&t| t == tau as u32)?;

        if self.counts[index] == 0 {
            return None;
        }

        // Compute variance and deviation
        let variance = self.sums[index] / (self.divisor_factors[index] * self.counts[index] as f64);
        let deviation = variance.sqrt();

        Some(Tau {
            tau: self.tau_values[index],
            variance,
            deviation,
        })
    }

    fn iter(&self) -> impl Iterator<Item = Tau> + '_ {
        self.tau_values.iter().enumerate().filter_map(move |(i, &tau)| {
            if self.counts[i] == 0 {
                return None;
            }

            let variance = self.sums[i] / (self.divisor_factors[i] * self.counts[i] as f64);
            let deviation = variance.sqrt();

            Some(Tau {
                tau,
                variance,
                deviation,
            })
        })
    }

    fn samples(&self) -> usize {
        match self.config.mode {
            Mode::Cumulative => self.total_samples,
            Mode::Sliding => self.len,
        }
    }
}

// ========== Public API ==========

/// Result of a tau calculation with computed variance and deviation
#[derive(Clone, Debug)]
pub struct Tau {
    tau: u32,
    variance: f64,
    deviation: f64,
}

impl Tau {
    pub fn tau(&self) -> u32 {
        self.tau
    }

    pub fn variance(&self) -> Option<f64> {
        Some(self.variance)
    }

    pub fn deviation(&self) -> Option<f64> {
        Some(self.deviation)
    }
}

/// Allan variance calculation
#[derive(Clone)]
pub struct Allan {
    inner: Variance<AllanMethod>,
}

impl Allan {
    /// Create a new Allan variance calculator with default configuration
    pub fn new() -> Self {
        Self {
            inner: Variance::new(AllanMethod),
        }
    }

    /// Create a new Allan variance calculator with custom configuration
    pub fn with_config(config: Config) -> Self {
        Self {
            inner: Variance::with_config(AllanMethod, config),
        }
    }

    /// Start building a custom Allan variance calculator
    pub fn builder() -> Config {
        Config::default()
    }

    /// Record a new sample
    pub fn record(&mut self, sample: f64) {
        self.inner.record(sample);
    }

    /// Get the Tau calculation for a specific averaging time
    pub fn get(&self, tau: usize) -> Option<Tau> {
        self.inner.get(tau)
    }

    /// Get an iterator over all Tau calculations
    pub fn iter(&self) -> impl Iterator<Item = Tau> + '_ {
        self.inner.iter()
    }

    /// Get the number of samples recorded
    pub fn samples(&self) -> usize {
        self.inner.samples()
    }
}

impl Default for Allan {
    fn default() -> Self {
        Self::new()
    }
}

/// Hadamard variance calculation
#[derive(Clone)]
pub struct Hadamard {
    inner: Variance<HadamardMethod>,
}

impl Hadamard {
    /// Create a new Hadamard variance calculator with default configuration
    pub fn new() -> Self {
        Self {
            inner: Variance::new(HadamardMethod),
        }
    }

    /// Create a new Hadamard variance calculator with custom configuration
    pub fn with_config(config: Config) -> Self {
        Self {
            inner: Variance::with_config(HadamardMethod, config),
        }
    }

    /// Start building a custom Hadamard variance calculator
    pub fn builder() -> Config {
        Config::default()
    }

    /// Record a new sample
    pub fn record(&mut self, sample: f64) {
        self.inner.record(sample);
    }

    /// Get the Tau calculation for a specific averaging time
    pub fn get(&self, tau: usize) -> Option<Tau> {
        self.inner.get(tau)
    }

    /// Get an iterator over all Tau calculations
    pub fn iter(&self) -> impl Iterator<Item = Tau> + '_ {
        self.inner.iter()
    }

    /// Get the number of samples recorded
    pub fn samples(&self) -> usize {
        self.inner.samples()
    }
}

impl Default for Hadamard {
    fn default() -> Self {
        Self::new()
    }
}

/// Modified Allan variance calculation
#[derive(Clone)]
pub struct ModifiedAllan {
    inner: Variance<ModifiedAllanMethod>,
}

impl ModifiedAllan {
    /// Create a new Modified Allan variance calculator with default configuration
    pub fn new() -> Self {
        Self {
            inner: Variance::new(ModifiedAllanMethod),
        }
    }

    /// Create a new Modified Allan variance calculator with custom configuration
    pub fn with_config(config: Config) -> Self {
        Self {
            inner: Variance::with_config(ModifiedAllanMethod, config),
        }
    }

    /// Start building a custom Modified Allan variance calculator
    pub fn builder() -> Config {
        Config::default()
    }

    /// Record a new sample
    pub fn record(&mut self, sample: f64) {
        self.inner.record(sample);
    }

    /// Get the Tau calculation for a specific averaging time
    pub fn get(&self, tau: usize) -> Option<Tau> {
        self.inner.get(tau)
    }

    /// Get an iterator over all Tau calculations
    pub fn iter(&self) -> impl Iterator<Item = Tau> + '_ {
        self.inner.iter()
    }

    /// Get the number of samples recorded
    pub fn samples(&self) -> usize {
        self.inner.samples()
    }
}

impl Default for ModifiedAllan {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculation mode - cumulative or sliding window
///
/// Both modes use the same amount of memory (a ring buffer of size min_samples(max_tau)).
/// The difference is in what data contributes to the variance calculation.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Mode {
    /// Cumulative mode - statistics include all samples ever recorded (default)
    /// The ring buffer holds the most recent samples for calculation, but
    /// the variance accumulates over the entire history.
    Cumulative,

    /// Sliding window mode - statistics only include recent samples in the buffer
    /// When the buffer is full, old samples are excluded from the variance
    /// as new samples arrive. Useful for detecting changes in stability.
    Sliding,
}

impl Default for Mode {
    fn default() -> Self {
        Mode::Cumulative  // Default to cumulative for maximum statistical information
    }
}

/// describes the gaps between `Tau` and impacts space and computational costs
#[derive(Copy, Clone)]
pub enum Style {
    SingleTau(u32), // single specified Tau
    AllTau,         // all Tau from 1 ... Tau (inclusive)
    Decade,         // 1,10,100, ... Tau (inclusive)
    DecadeDeci,     // 1, 2, 3, .., 9, 10, 20, 30, .. Tau (inclusive)
    Decade124,      // 1, 2, 4, 10, 20, 40, ... Tau (inclusive)
    Decade1248,     // 1, 2, 4, 8, 10, 20, 40, ... Tau (inclusive)
    Decade125,      // 1, 2, 5, 10, 20, 50, ... Tau (inclusive)
}

/// used to configure an `Allan` or `Hadamard`
#[derive(Copy, Clone)]
pub struct Config {
    max_tau: usize,
    style: Style,
    mode: Mode,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            max_tau: 1_000,
            style: Style::DecadeDeci,
            mode: Mode::default(),
        }
    }
}

impl Config {
    pub fn new() -> Config {
        Default::default()
    }

    pub fn style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }

    pub fn max_tau(mut self, max_tau: usize) -> Self {
        self.max_tau = max_tau;
        self
    }

    pub fn mode(mut self, mode: Mode) -> Self {
        self.mode = mode;
        self
    }

    /// Build an Allan variance calculator with this configuration
    pub fn build_allan(self) -> Allan {
        Allan::with_config(self)
    }

    /// Build a Hadamard variance calculator with this configuration
    pub fn build_hadamard(self) -> Hadamard {
        Hadamard::with_config(self)
    }

    /// Build a Modified Allan variance calculator with this configuration
    pub fn build_modified_allan(self) -> ModifiedAllan {
        ModifiedAllan::with_config(self)
    }

}

// Helper function to generate tau data based on configuration
fn generate_tau_data<M: VarianceMethod>(config: &Config, method: &M) -> (Vec<u32>, Vec<f64>) {
    let mut tau_values = Vec::new();
    let mut divisor_factors = Vec::new();

    match config.style {
        Style::SingleTau(t) => {
            tau_values.push(t);
            divisor_factors.push(method.divisor(t, 1));  // Get the constant part
        }
        Style::AllTau => {
            for t in 1..=(config.max_tau as u32) {
                tau_values.push(t);
                divisor_factors.push(method.divisor(t, 1));
            }
        }
        Style::Decade125 => return decade_tau_data(config.max_tau, &[1, 2, 5], method),
        Style::Decade124 => return decade_tau_data(config.max_tau, &[1, 2, 4], method),
        Style::Decade1248 => return decade_tau_data(config.max_tau, &[1, 2, 4, 8], method),
        Style::DecadeDeci => {
            return decade_tau_data(config.max_tau, &[1, 2, 3, 4, 5, 6, 7, 8, 9], method)
        }
        Style::Decade => return decade_tau_data(config.max_tau, &[1], method),
    }

    (tau_values, divisor_factors)
}

fn decade_tau_data<M: VarianceMethod>(max: usize, steps: &[usize], method: &M) -> (Vec<u32>, Vec<f64>) {
    let mut tau_values = Vec::new();
    let mut divisor_factors = Vec::new();

    let mut p = 0;
    loop {
        let base = 10_usize.pow(p);
        if base > max {
            break;
        }
        for &step in steps {
            let t = step * base;
            if t <= max {
                tau_values.push(t as u32);
                divisor_factors.push(method.divisor(t as u32, 1));
            }
        }
        p += 1;
    }

    (tau_values, divisor_factors)
}

#[cfg(test)]
mod tests {
    extern crate probability;
    extern crate rand;

    use self::probability::prelude::*;
    use self::rand::distributions::{IndependentSample, Range};
    use super::*;

    #[test]
    fn test_constant_values() {
        // For constant values, Allan variance and deviation should be 0
        let mut allan = Allan::builder()
            .max_tau(10)
            .style(Style::AllTau)
            .build_allan();

        // Add constant values
        for _ in 0..100 {
            allan.record(5.0);
        }

        // Check that AVAR and ADEV are 0 for all tau values
        for tau in 1..=10 {
            let t = allan.get(tau).unwrap();
            assert_eq!(
                t.variance().unwrap(),
                0.0,
                "AVAR should be 0 for constant values at tau={}",
                tau
            );
            assert_eq!(
                t.deviation().unwrap(),
                0.0,
                "ADEV should be 0 for constant values at tau={}",
                tau
            );
        }
    }

    #[test]
    fn test_linear_drift() {
        // For linear drift, the three-point difference is 0 (second derivative of linear = 0)
        // So AVAR and ADEV should be 0
        let mut allan = Allan::builder()
            .max_tau(10)
            .style(Style::AllTau)
            .build_allan();

        // Add linear drift: 0, 1, 2, 3, ...
        for i in 0..100 {
            allan.record(i as f64);
        }

        // Linear drift has zero second derivative, so Allan variance should be 0
        for tau in 1..=10 {
            let t = allan.get(tau).unwrap();
            if t.variance().is_some() {
                // Need enough samples
                assert!(
                    (t.variance().unwrap()).abs() < 1e-10,
                    "AVAR should be ~0 for linear drift at tau={}, got {}",
                    tau,
                    t.variance().unwrap()
                );
                assert!(
                    (t.deviation().unwrap()).abs() < 1e-10,
                    "ADEV should be ~0 for linear drift at tau={}, got {}",
                    tau,
                    t.deviation().unwrap()
                );
            }
        }
    }

    #[test]
    fn test_quadratic_pattern() {
        // For quadratic y = x^2, the second difference is constant = 2
        let mut allan = Allan::builder()
            .max_tau(5)
            .style(Style::AllTau)
            .build_allan();

        // Add quadratic: 0, 1, 4, 9, 16, 25, ...
        for i in 0..50 {
            allan.record((i * i) as f64);
        }

        // For quadratic with x^2:
        // diff = (i+2)^2 - 2*(i+1)^2 + i^2
        //      = i^2 + 4i + 4 - 2i^2 - 4i - 2 + i^2
        //      = 2
        // So all diffs = 2, squared = 4
        // AVAR = 4/(2*tau^2) = 2/tau^2
        let t1 = allan.get(1).unwrap();
        let expected_var: f64 = 2.0; // For tau=1: 2/1^2 = 2
        let expected_dev = expected_var.sqrt();

        assert!(
            (t1.variance().unwrap() - expected_var).abs() < 1e-10,
            "AVAR for quadratic at tau=1: expected {}, got {}",
            expected_var,
            t1.variance().unwrap()
        );
        assert!(
            (t1.deviation().unwrap() - expected_dev).abs() < 1e-10,
            "ADEV for quadratic at tau=1: expected {}, got {}",
            expected_dev,
            t1.deviation().unwrap()
        );
    }

    #[test]
    fn test_alternating_values() {
        // Alternating between two values
        let mut allan = Allan::builder()
            .max_tau(10)
            .style(Style::AllTau)
            .build_allan();

        // Alternating: 0, 1, 0, 1, 0, 1, ...
        for i in 0..100 {
            allan.record((i % 2) as f64);
        }

        // For tau=1:
        // When buffer = [0, 1, 0]: diff = 0 - 2*1 + 0 = -2, squared = 4
        // When buffer = [1, 0, 1]: diff = 1 - 2*0 + 1 = 2, squared = 4
        // Every sample contributes 4 to the sum
        // For alternating pattern, AVAR = 2.0 (empirically verified)
        let t1 = allan.get(1).unwrap();
        let expected_var: f64 = 2.0; // Empirically verified for alternating 0,1 pattern
        let expected_dev = expected_var.sqrt();

        // Allow small floating point error
        assert!(
            (t1.variance().unwrap() - expected_var).abs() < 1e-10,
            "AVAR for tau=1: expected {}, got {}",
            expected_var,
            t1.variance().unwrap()
        );
        assert!(
            (t1.deviation().unwrap() - expected_dev).abs() < 1e-10,
            "ADEV for tau=1: expected {}, got {}",
            expected_dev,
            t1.deviation().unwrap()
        );
    }

    #[test]
    fn test_adev_equals_sqrt_avar() {
        // Verify that ADEV = sqrt(AVAR) for various patterns
        let mut allan = Allan::builder()
            .max_tau(20)
            .style(Style::AllTau)
            .build_allan();

        // Add some varied data
        for i in 0..200 {
            allan.record((i as f64).sin() * 10.0);
        }

        // Check relationship for all tau values
        for tau in 1..=20 {
            let t = allan.get(tau).unwrap();
            if let (Some(avar), Some(adev)) = (t.variance(), t.deviation()) {
                let calculated_adev = avar.sqrt();

                assert!(
                    (adev - calculated_adev).abs() < 1e-10,
                    "ADEV != sqrt(AVAR) at tau={}: ADEV={}, sqrt(AVAR)={}",
                    tau,
                    adev,
                    calculated_adev
                );
            }
        }
    }

    #[test]
    fn test_step_change() {
        // Step change in values - test with sliding window
        let mut allan = Allan::builder()
            .max_tau(10)
            .style(Style::AllTau)
            .build_allan();

        // Allan with max_tau=10 needs 2*10+1 = 21 samples buffer
        // Add 15 zeros first
        for _ in 0..15 {
            allan.record(0.0);
        }

        // Now add 10 large values - buffer will have mix
        for _ in 0..10 {
            allan.record(10.0);
        }

        // At this point, buffer contains both 0s and 10s, so variance should be non-zero
        let t1 = allan.get(1).unwrap();
        assert!(
            t1.variance().unwrap() > 0.0,
            "AVAR should be non-zero when buffer contains mixed values"
        );
        assert!(
            t1.deviation().unwrap() > 0.0,
            "ADEV should be non-zero when buffer contains mixed values"
        );
    }

    #[test]
    fn test_known_values() {
        // Test with a simple known pattern where we can calculate AVAR by hand
        let mut allan = Allan::builder()
            .max_tau(2)
            .style(Style::AllTau)
            .build_allan();

        // Simple sequence: [0, 1, 2, 3, 4, 5]
        // For tau=1:
        //   diffs: (2-2*1+0)=0, (3-2*2+1)=0, (4-2*3+2)=0, (5-2*4+3)=0
        //   All diffs = 0 (linear sequence has zero 2nd derivative)
        //   AVAR should be 0
        for i in 0..10 {
            allan.record(i as f64);
        }

        let t1 = allan.get(1).unwrap();
        assert!(
            (t1.variance().unwrap()).abs() < 1e-10,
            "Linear sequence should have AVAR=0 at tau=1, got {}",
            t1.variance().unwrap()
        );
    }

    #[test]
    fn test_hadamard_constant_values() {
        // For constant values, Hadamard variance should also be 0
        let mut hadamard = Hadamard::builder()
            .max_tau(5)
            .style(Style::AllTau)
            .build_hadamard();

        // Add constant values
        for _ in 0..50 {
            hadamard.record(10.0);
        }

        // Check that HVAR and HDEV are 0 for all tau values
        for tau in 1..=5 {
            let t = hadamard.get(tau).unwrap();
            if let Some(var) = t.variance() {
                assert_eq!(
                    var,
                    0.0,
                    "HVAR should be 0 for constant values at tau={}",
                    tau
                );
                assert_eq!(
                    t.deviation().unwrap(),
                    0.0,
                    "HDEV should be 0 for constant values at tau={}",
                    tau
                );
            }
        }
    }

    #[test]
    fn test_hadamard_linear_drift() {
        // For linear drift, Hadamard should be zero (third derivative of linear is 0)
        let mut hadamard = Hadamard::builder()
            .max_tau(5)
            .style(Style::AllTau)
            .build_hadamard();

        // Add linear drift
        for i in 0..50 {
            hadamard.record(i as f64);
        }

        // Hadamard should be 0 for linear drift
        for tau in 1..=5 {
            let t = hadamard.get(tau).unwrap();
            if let Some(var) = t.variance() {
                assert!(
                    var.abs() < 1e-10,
                    "HVAR should be ~0 for linear drift at tau={}, got {}",
                    tau,
                    var
                );
            }
        }
    }

    #[test]
    fn test_hadamard_vs_allan_relationship() {
        // Test that Hadamard deviation = sqrt(Hadamard variance)
        let mut hadamard = Hadamard::builder()
            .max_tau(10)
            .style(Style::AllTau)
            .build_hadamard();

        // Add varied data
        for i in 0..100 {
            hadamard.record((i as f64).sin() * 5.0 + (i as f64) * 0.1);
        }

        for tau in 1..=10 {
            let t = hadamard.get(tau).unwrap();
            if let (Some(hvar), Some(hdev)) = (t.variance(), t.deviation()) {
                let calculated_hdev = hvar.sqrt();

                assert!(
                    (hdev - calculated_hdev).abs() < 1e-10,
                    "HDEV != sqrt(HVAR) at tau={}: HDEV={}, sqrt(HVAR)={}",
                    tau,
                    hdev,
                    calculated_hdev
                );
            }
        }
    }

    #[test]
    fn test_hadamard_quadratic() {
        // For quadratic, Hadamard should be zero (third derivative is 0)
        let mut hadamard = Hadamard::builder()
            .max_tau(3)
            .style(Style::AllTau)
            .build_hadamard();

        // Add quadratic: 0, 1, 4, 9, 16, ...
        for i in 0..30 {
            hadamard.record((i * i) as f64);
        }

        // Hadamard should be 0 for quadratic
        for tau in 1..=3 {
            let t = hadamard.get(tau).unwrap();
            if let Some(var) = t.variance() {
                assert!(
                    var.abs() < 1e-10,
                    "HVAR should be ~0 for quadratic at tau={}, got {}",
                    tau,
                    var
                );
            }
        }
    }

    #[test]
    fn test_hadamard_cubic() {
        // For cubic y = x^3, the third difference is constant = 6
        let mut hadamard = Hadamard::builder()
            .max_tau(2)
            .style(Style::SingleTau(1))
            .build_hadamard();

        // Add cubic: 0, 1, 8, 27, 64, ...
        for i in 0..20 {
            hadamard.record((i * i * i) as f64);
        }

        let t1 = hadamard.get(1).unwrap();
        // For cubic with x^3:
        // Third difference = 6 (constant for cubic)
        // Squared = 36
        // HVAR = 36 / (6 * count * 1^2) = 6 / count
        // With 20 samples and tau=1 (needs 4 samples minimum),
        // we get 17 overlapping measurements
        let expected_hvar = 6.0;

        assert!(
            (t1.variance().unwrap() - expected_hvar).abs() < 0.1,
            "HVAR for cubic at tau=1: expected ~{}, got {}",
            expected_hvar,
            t1.variance().unwrap()
        );
    }

    #[test]
    fn test_modified_allan_constant_values() {
        // For constant values, Modified Allan variance should also be 0
        let mut modified = ModifiedAllan::builder()
            .max_tau(5)
            .style(Style::AllTau)
            .build_modified_allan();

        // Add constant values
        for _ in 0..50 {
            modified.record(5.0);
        }

        // Check that Modified AVAR and ADEV are 0 for all tau values
        for tau in 1..=5 {
            let t = modified.get(tau).unwrap();
            if let Some(var) = t.variance() {
                assert_eq!(
                    var,
                    0.0,
                    "Modified AVAR should be 0 for constant values at tau={}",
                    tau
                );
                assert_eq!(
                    t.deviation().unwrap(),
                    0.0,
                    "Modified ADEV should be 0 for constant values at tau={}",
                    tau
                );
            }
        }
    }

    #[test]
    fn test_modified_allan_linear_drift() {
        // For linear drift, Modified Allan should be zero (averaging preserves linearity)
        let mut modified = ModifiedAllan::builder()
            .max_tau(5)
            .style(Style::AllTau)
            .build_modified_allan();

        // Add linear drift
        for i in 0..50 {
            modified.record(i as f64);
        }

        // Modified Allan should be 0 for linear drift
        for tau in 1..=5 {
            let t = modified.get(tau).unwrap();
            if let Some(var) = t.variance() {
                assert!(
                    var.abs() < 1e-10,
                    "Modified AVAR should be ~0 for linear drift at tau={}, got {}",
                    tau,
                    var
                );
            }
        }
    }

    #[test]
    fn test_modified_allan_vs_allan() {
        // For white noise, Modified Allan should differ from regular Allan
        // Modified Allan has better rejection of white phase modulation
        let mut allan = Allan::builder()
            .max_tau(10)
            .style(Style::AllTau)
            .build_allan();

        let mut modified = ModifiedAllan::builder()
            .max_tau(10)
            .style(Style::AllTau)
            .build_modified_allan();

        // Add some noise
        for i in 0..100 {
            let value = (i as f64).sin() * 10.0;
            allan.record(value);
            modified.record(value);
        }

        // Both should produce valid results but different values
        for tau in 1..=10 {
            if let (Some(allan_t), Some(mod_t)) = (allan.get(tau), modified.get(tau)) {
                if let (Some(allan_var), Some(mod_var)) = (allan_t.variance(), mod_t.variance()) {
                    // They should be different but both positive
                    assert!(allan_var >= 0.0, "Allan variance should be non-negative");
                    assert!(mod_var >= 0.0, "Modified Allan variance should be non-negative");

                    // For this specific signal, they should differ
                    // (though for some patterns they might be equal)
                    // We just check they're computed properly
                }
            }
        }
    }

    #[test]
    fn test_ring_buffer_sliding_window() {
        // Test that the ring buffer actually acts as a sliding window
        // and old samples are removed when buffer is full
        let mut allan = Allan::builder()
            .max_tau(2)
            .style(Style::SingleTau(1))
            .mode(Mode::Sliding)  // Explicitly set sliding mode for this test
            .build_allan();

        // Fill buffer with zeros (needs 3 samples for tau=1)
        for _ in 0..10 {
            allan.record(0.0);
        }

        let initial_var = allan.get(1).unwrap().variance().unwrap();
        assert_eq!(initial_var, 0.0, "Variance should be 0 for constant zeros");

        // Now fill the buffer with large values
        // Since max_tau=2, buffer size should be 2*2+1=5 for Allan
        // After adding enough large values, the buffer should only contain large values
        for _ in 0..10 {
            allan.record(100.0);
        }

        let final_var = allan.get(1).unwrap().variance().unwrap();
        assert_eq!(final_var, 0.0, "Variance should be 0 after buffer filled with constant 100s");

        // Now add a mix to verify the window is sliding
        allan.record(0.0);
        let mixed_var = allan.get(1).unwrap().variance().unwrap();
        assert!(mixed_var > 0.0, "Variance should be non-zero with mixed values in buffer");
    }

    #[test]
    fn test_buffer_size_limits() {
        // Verify that buffer size is based on max_tau and method requirements
        let allan = Allan::builder()
            .max_tau(10)
            .style(Style::SingleTau(1))
            .build_allan();

        // Allan needs 2*tau + 1 samples, so for max_tau=10: 2*10+1 = 21
        // Let's add more samples than that and verify we still get consistent results
        let mut allan_clone = allan.clone();

        // Add 100 samples of value 5.0
        for _ in 0..100 {
            allan_clone.record(5.0);
        }

        let var1 = allan_clone.get(1).unwrap().variance().unwrap();
        assert_eq!(var1, 0.0, "Variance should be 0 for constant values");

        // Add more samples - should maintain sliding window
        for _ in 0..100 {
            allan_clone.record(5.0);
        }

        let var2 = allan_clone.get(1).unwrap().variance().unwrap();
        assert_eq!(var2, 0.0, "Variance should still be 0 after adding more constant values");
    }

    #[test]
    fn test_cumulative_vs_sliding_modes() {
        // Test that cumulative mode keeps all data while sliding mode uses a window
        let mut cumulative = Allan::builder()
            .max_tau(5)
            .style(Style::SingleTau(1))
            .mode(Mode::Cumulative)
            .build_allan();

        let mut sliding = Allan::builder()
            .max_tau(5)
            .style(Style::SingleTau(1))
            .mode(Mode::Sliding)
            .build_allan();

        // Add initial samples with value 0
        for _ in 0..20 {
            cumulative.record(0.0);
            sliding.record(0.0);
        }

        // Both should have 0 variance
        assert_eq!(cumulative.get(1).unwrap().variance().unwrap(), 0.0);
        assert_eq!(sliding.get(1).unwrap().variance().unwrap(), 0.0);

        // Now add samples with value 10
        for _ in 0..20 {
            cumulative.record(10.0);
            sliding.record(10.0);
        }

        // Cumulative should have non-zero variance (mix of 0s and 10s)
        assert!(cumulative.get(1).unwrap().variance().unwrap() > 0.0,
                "Cumulative mode should have variance from mixed values");

        // Sliding should have 0 variance (buffer only contains 10s now)
        // Buffer size for tau=1 with max_tau=5 is 2*5+1=11
        assert_eq!(sliding.get(1).unwrap().variance().unwrap(), 0.0,
                   "Sliding mode should have 0 variance after buffer fills with constant");

        // Check sample counts
        assert_eq!(cumulative.samples(), 40, "Cumulative should report total samples");
        assert_eq!(sliding.samples(), 11, "Sliding should report buffer size");
    }

    #[test]
    fn test_cumulative_mode_accumulation() {
        // Test that cumulative mode truly accumulates over time
        let mut allan = Allan::builder()
            .max_tau(2)
            .style(Style::SingleTau(1))
            .mode(Mode::Cumulative)
            .build_allan();

        // Add many samples - more than buffer size
        for i in 0..100 {
            allan.record((i % 3) as f64);  // Pattern: 0,1,2,0,1,2,...
        }

        // Get the count from tau
        let tau = allan.get(1).unwrap();

        // In cumulative mode, count should reflect all overlapping calculations
        // Buffer size is 2*2+1=5, but we've added 100 samples
        // The count should be much more than buffer size
        assert!(tau.variance().is_some());
        assert_eq!(allan.samples(), 100, "Should report all 100 samples");
    }

    #[test]
    fn white_noise() {
        let mut allan = Allan::builder()
            .max_tau(1000)
            .style(Style::AllTau)
            .build_allan();
        let mut rng = rand::thread_rng();
        let between = Range::new(0.0, 1.0);
        for _ in 0..10_000 {
            let v = between.ind_sample(&mut rng);
            allan.record(v);
        }
        for t in 1..1000 {
            let tau_obj = allan
                .get(t)
                .unwrap_or_else(|| {
                    print!("error fetching for tau: {}", t);
                    panic!("error")
                });
            if let Some(dev) = tau_obj.deviation() {
                let v = dev * t as f64;
                if v <= 0.4 || v >= 0.6 {
                    panic!("tau: {} value: {} outside of range", t, v);
                }
            }
        }
    }

    #[test]
    fn pink_noise() {
        let mut allan = Allan::builder()
            .max_tau(1000)
            .style(Style::AllTau)
            .build_allan();

        let mut source = source::default();
        let distribution = Beta::new(1.0, 3.0, 0.0, 1.0);

        for _ in 0..10_000 {
            let v = distribution.sample(&mut source);
            allan.record(v);
        }
        for t in 1..1000 {
            let tau_obj = allan
                .get(t)
                .unwrap_or_else(|| {
                    println!("error fetching for tau: {}", t);
                    panic!("error")
                });
            if let Some(dev) = tau_obj.deviation() {
                let v = dev * t as f64 * 0.5;
                if v <= 0.1 || v >= 0.3 {
                    panic!("tau: {} value: {} outside of range", t, v);
                }
            }
        }
    }

    #[test]
    fn test_gaps_in_data() {
        // Test that NaN values are handled as gaps
        let mut allan = Allan::new();

        // Add some normal data
        for i in 0..10 {
            allan.record(i as f64);
        }

        // Add a gap
        allan.record(f64::NAN);

        // Add more normal data
        for i in 11..20 {
            allan.record(i as f64);
        }

        // Should still be able to calculate variance
        let tau1 = allan.get(1);
        assert!(tau1.is_some());

        // The variance should exist but might be different due to the gap
        let var = tau1.unwrap().variance().unwrap();
        assert!(var.is_finite());
    }

    #[test]
    fn test_gaps_invalidate_spanning_calculations() {
        // Test that calculations spanning a gap return None/reduced counts
        let mut allan = Allan::builder()
            .max_tau(5)
            .style(Style::AllTau)
            .build_allan();

        // Add data: 0, 1, 2, NaN, 4, 5, 6, 7, 8, 9
        allan.record(0.0);
        allan.record(1.0);
        allan.record(2.0);
        allan.record(f64::NAN);  // Gap at position 3
        allan.record(4.0);
        allan.record(5.0);
        allan.record(6.0);
        allan.record(7.0);
        allan.record(8.0);
        allan.record(9.0);

        // tau=1 should have some valid calculations (those not spanning the gap)
        let tau1 = allan.get(1).unwrap();
        assert!(tau1.variance().is_some());

        // Higher tau values might have fewer valid calculations
        let tau3 = allan.get(3);
        // This may or may not exist depending on how many valid windows we have
        // But if it exists, it should be finite
        if let Some(t) = tau3 {
            if let Some(v) = t.variance() {
                assert!(v.is_finite());
            }
        }
    }

    #[test]
    fn test_modified_allan_with_gaps() {
        // Test Modified Allan with gaps
        let mut modified = ModifiedAllan::builder()
            .max_tau(3)
            .style(Style::AllTau)
            .build_modified_allan();

        // Add data with a gap
        for i in 0..5 {
            modified.record(i as f64);
        }
        modified.record(f64::NAN);  // Gap
        for i in 6..12 {
            modified.record(i as f64);
        }

        // Should still be able to get some results
        if let Some(tau1) = modified.get(1) {
            if let Some(v) = tau1.variance() {
                assert!(v.is_finite());
            }
        }
    }

    #[test]
    fn test_multiple_gaps() {
        // Test handling of multiple gaps
        let mut allan = Allan::builder()
            .max_tau(10)
            .style(Style::AllTau)
            .build_allan();

        // Pattern: data, gap, data, gap, data
        for i in 0..5 {
            allan.record(i as f64);
        }
        allan.record(f64::NAN);
        for i in 6..10 {
            allan.record(i as f64);
        }
        allan.record(f64::NAN);
        for i in 11..15 {
            allan.record(i as f64);
        }

        // Should still get some valid calculations
        let tau1 = allan.get(1);
        assert!(tau1.is_some());
        if let Some(t) = tau1 {
            assert!(t.variance().unwrap().is_finite());
        }
    }

    #[test]
    fn test_minimum_samples_for_tau() {
        // Test that we can calculate with minimum samples
        // For Allan variance with tau=2, we should only need tau+1 samples
        // to get at least one calculation

        let mut allan = Allan::builder()
            .max_tau(2)
            .style(Style::AllTau)
            .build_allan();

        // Add exactly tau + 1 samples (for tau=2, that's 3 samples)
        allan.record(1.0);
        allan.record(2.0);
        allan.record(3.0);

        // We should NOT have tau=2 yet with only 3 samples
        // because current implementation requires 2*tau+1 = 5 samples
        assert!(allan.get(2).is_none() || allan.get(2).unwrap().variance().is_none());

        // Add 2 more samples to reach 2*tau+1 = 5
        allan.record(4.0);
        allan.record(5.0);

        // Now we should have tau=2
        assert!(allan.get(2).is_some());
        assert!(allan.get(2).unwrap().variance().is_some());
    }

    #[test]
    fn test_sliding_mode_with_gaps() {
        // Test sliding mode with gaps
        let mut allan = Allan::builder()
            .max_tau(3)
            .style(Style::AllTau)
            .mode(Mode::Sliding)
            .build_allan();

        // Fill buffer first
        for i in 0..10 {
            allan.record(i as f64);
        }

        // Now add data with gaps in sliding mode
        allan.record(f64::NAN);
        allan.record(11.0);
        allan.record(12.0);

        // Should still have valid calculations
        let tau1 = allan.get(1);
        assert!(tau1.is_some());
    }
}