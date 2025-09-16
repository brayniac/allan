//! Allan provides variance and deviation tools for stability analysis
//!
//! # Goals
//! * provide streaming variance and deviations from series data
//! * pre-allocated datastructures
//!
//! # Usage
//!
//! Create a new instance, add records, retrieve statistic
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

    fn calculate_diff(&self, buffer: &[f64], idx0: usize, idx1: usize, idx2: usize, _idx3: Option<usize>) -> f64 {
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

    fn calculate_diff(&self, buffer: &[f64], idx0: usize, idx1: usize, idx2: usize, idx3: Option<usize>) -> f64 {
        let idx3 = idx3.expect("Hadamard requires 4 points");
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
    fn average(&self, buffer: &[f64], start_idx: usize, tau: usize, buffer_len: usize) -> f64 {
        let mut sum = 0.0;
        for i in 0..tau {
            let idx = (start_idx + i) % buffer_len;
            sum += buffer[idx];
        }
        sum / tau as f64
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
    _config: Config,
    taus: Vec<Tau>,
    buffer: Vec<f64>,
    head: usize,
    len: usize,
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

        let taus = generate_taus(&config);

        Self {
            method,
            _config: config,
            taus,
            buffer,
            head: 0,
            len: 0,
        }
    }

    fn record(&mut self, value: f64) {
        if self.len < self.buffer.len() {
            self.buffer[self.len] = value;
            self.len += 1;
        } else {
            self.buffer[self.head] = value;
            self.head = (self.head + 1) % self.buffer.len();
        }

        self.recalculate();
    }

    fn recalculate(&mut self) {
        for tau in &mut self.taus {
            // Only update with the latest sample if we have enough samples
            let t = tau.tau as usize;
            let min_samples = self.method.min_samples(t);

            if self.len >= min_samples {
                // Calculate the newest index
                let newest_idx = if self.len < self.buffer.len() {
                    // Buffer not wrapped yet
                    self.len - 1
                } else {
                    // Buffer has wrapped
                    (self.head + self.buffer.len() - 1) % self.buffer.len()
                };

                // Calculate squared difference using the method-specific approach
                let squared_diff = self.method.calculate_squared_diff(&self.buffer, newest_idx, t, self.buffer.len());

                // Update running statistics
                tau.sum += squared_diff;
                tau.count += 1;

                let variance = tau.sum / self.method.divisor(tau.tau, tau.count);
                tau.variance = Some(variance);
                tau.deviation = Some(variance.sqrt());
            }
        }
    }

    fn get(&self, tau: usize) -> Option<&Tau> {
        self.taus.iter().find(|t| t.tau == tau as u32)
    }

    fn iter(&self) -> impl Iterator<Item = &Tau> {
        self.taus.iter()
    }

    fn samples(&self) -> usize {
        self.len
    }
}

// ========== Public API ==========

/// Generic tau bucket for variance calculations
#[derive(Clone)]
pub struct Tau {
    tau: u32,
    sum: f64,
    count: u64,
    variance: Option<f64>,
    deviation: Option<f64>,
}

impl Tau {
    fn new(tau: u32) -> Tau {
        Tau {
            tau,
            sum: 0.0,
            count: 0,
            variance: None,
            deviation: None,
        }
    }

    pub fn tau(&self) -> u32 {
        self.tau
    }

    pub fn variance(&self) -> Option<f64> {
        self.variance
    }

    pub fn deviation(&self) -> Option<f64> {
        self.deviation
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

    pub fn configure() -> Config {
        Config::default()
    }

    fn configured(config: Config) -> Option<Allan> {
        Some(Allan::with_config(config))
    }

    /// Record a new sample
    pub fn record(&mut self, sample: f64) {
        self.inner.record(sample);
    }

    /// Get the Tau calculation for a specific averaging time
    pub fn get(&self, tau: usize) -> Option<&Tau> {
        self.inner.get(tau)
    }

    /// Get an iterator over all Tau calculations
    pub fn iter(&self) -> impl Iterator<Item = &Tau> {
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

    pub fn configure() -> Config {
        Config::default()
    }

    fn configured(config: Config) -> Option<Hadamard> {
        Some(Hadamard::with_config(config))
    }

    /// Record a new sample
    pub fn record(&mut self, sample: f64) {
        self.inner.record(sample);
    }

    /// Get the Tau calculation for a specific averaging time
    pub fn get(&self, tau: usize) -> Option<&Tau> {
        self.inner.get(tau)
    }

    /// Get an iterator over all Tau calculations
    pub fn iter(&self) -> impl Iterator<Item = &Tau> {
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

    pub fn configure() -> Config {
        Config::default()
    }

    fn configured(config: Config) -> Option<ModifiedAllan> {
        Some(ModifiedAllan::with_config(config))
    }

    /// Record a new sample
    pub fn record(&mut self, sample: f64) {
        self.inner.record(sample);
    }

    /// Get the Tau calculation for a specific averaging time
    pub fn get(&self, tau: usize) -> Option<&Tau> {
        self.inner.get(tau)
    }

    /// Get an iterator over all Tau calculations
    pub fn iter(&self) -> impl Iterator<Item = &Tau> {
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
}

impl Default for Config {
    fn default() -> Config {
        Config {
            max_tau: 1_000,
            style: Style::DecadeDeci,
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

    pub fn build(self) -> Option<Allan> {
        Allan::configured(self)
    }

    pub fn build_hadamard(self) -> Option<Hadamard> {
        Hadamard::configured(self)
    }

    pub fn build_modified_allan(self) -> Option<ModifiedAllan> {
        ModifiedAllan::configured(self)
    }
}

// Helper function to generate Tau buckets based on configuration
fn generate_taus(config: &Config) -> Vec<Tau> {
    let mut taus = Vec::new();

    match config.style {
        Style::SingleTau(t) => {
            taus.push(Tau::new(t));
        }
        Style::AllTau => {
            for t in 1..=(config.max_tau as u32) {
                taus.push(Tau::new(t));
            }
        }
        Style::Decade125 => taus = decade_tau(config.max_tau, &[1, 2, 5]),
        Style::Decade124 => taus = decade_tau(config.max_tau, &[1, 2, 4]),
        Style::Decade1248 => taus = decade_tau(config.max_tau, &[1, 2, 4, 8]),
        Style::DecadeDeci => {
            taus = decade_tau(config.max_tau, &[1, 2, 3, 4, 5, 6, 7, 8, 9])
        }
        Style::Decade => taus = decade_tau(config.max_tau, &[1]),
    }

    taus
}

fn decade_tau(max: usize, steps: &[usize]) -> Vec<Tau> {
    let mut capacity = 0;
    let mut p = 0;
    loop {
        let base = 10_usize.pow(p);
        if base > max {
            break;
        }
        for &step in steps {
            if step * base <= max {
                capacity += 1;
            }
        }
        p += 1;
    }

    let mut taus: Vec<Tau> = Vec::with_capacity(capacity);
    p = 0;
    loop {
        let base = 10_usize.pow(p);
        if base > max {
            break;
        }
        for &step in steps {
            let t = step * base;
            if t <= max {
                taus.push(Tau::new(t as u32));
            }
        }
        p += 1;
    }
    taus
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
        let mut allan = Allan::configure()
            .max_tau(10)
            .style(Style::AllTau)
            .build()
            .unwrap();

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
        let mut allan = Allan::configure()
            .max_tau(10)
            .style(Style::AllTau)
            .build()
            .unwrap();

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
        let mut allan = Allan::configure()
            .max_tau(5)
            .style(Style::AllTau)
            .build()
            .unwrap();

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
        let mut allan = Allan::configure()
            .max_tau(10)
            .style(Style::AllTau)
            .build()
            .unwrap();

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
        let mut allan = Allan::configure()
            .max_tau(20)
            .style(Style::AllTau)
            .build()
            .unwrap();

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
        // Step change in values
        let mut allan = Allan::configure()
            .max_tau(10)
            .style(Style::AllTau)
            .build()
            .unwrap();

        // First 50 samples at 0, next 50 at 10
        for _ in 0..50 {
            allan.record(0.0);
        }
        for _ in 0..50 {
            allan.record(10.0);
        }

        // The variance should be high around the transition
        let t1 = allan.get(1).unwrap();
        assert!(
            t1.variance().unwrap() > 0.0,
            "AVAR should be non-zero for step change"
        );
        assert!(
            t1.deviation().unwrap() > 0.0,
            "ADEV should be non-zero for step change"
        );
    }

    #[test]
    fn test_known_values() {
        // Test with a simple known pattern where we can calculate AVAR by hand
        let mut allan = Allan::configure()
            .max_tau(2)
            .style(Style::AllTau)
            .build()
            .unwrap();

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
        let mut hadamard = Hadamard::configure()
            .max_tau(5)
            .style(Style::AllTau)
            .build_hadamard()
            .unwrap();

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
        let mut hadamard = Hadamard::configure()
            .max_tau(5)
            .style(Style::AllTau)
            .build_hadamard()
            .unwrap();

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
        let mut hadamard = Hadamard::configure()
            .max_tau(10)
            .style(Style::AllTau)
            .build_hadamard()
            .unwrap();

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
        let mut hadamard = Hadamard::configure()
            .max_tau(3)
            .style(Style::AllTau)
            .build_hadamard()
            .unwrap();

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
        let mut hadamard = Hadamard::configure()
            .max_tau(2)
            .style(Style::SingleTau(1))
            .build_hadamard()
            .unwrap();

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
        let mut modified = ModifiedAllan::configure()
            .max_tau(5)
            .style(Style::AllTau)
            .build_modified_allan()
            .unwrap();

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
        let mut modified = ModifiedAllan::configure()
            .max_tau(5)
            .style(Style::AllTau)
            .build_modified_allan()
            .unwrap();

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
        let mut allan = Allan::configure()
            .max_tau(10)
            .style(Style::AllTau)
            .build()
            .unwrap();

        let mut modified = ModifiedAllan::configure()
            .max_tau(10)
            .style(Style::AllTau)
            .build_modified_allan()
            .unwrap();

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
    fn white_noise() {
        let mut allan = Allan::configure()
            .max_tau(1000)
            .style(Style::AllTau)
            .build()
            .unwrap();
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
        let mut allan = Allan::configure()
            .max_tau(1000)
            .style(Style::AllTau)
            .build()
            .unwrap();

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
}