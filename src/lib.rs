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
//! // a default Allan
//! let mut allan = Allan::new();
//! for _ in 0..100 {
//!     allan.record(1.0);
//! }
//! assert_eq!(allan.get(1).unwrap().deviation().unwrap(), 0.0);
//!
//! // a configured Allan
//! let mut allan = Allan::configure().max_tau(10_000).build().unwrap();

/// the main datastructure for Allan
#[derive(Clone)]
pub struct Allan {
    samples: usize,
    config: Config,
    taus: Vec<Tau>,
    buffer: Vec<f64>,
    head: usize,  // Index where the next element will be written
    len: usize,   // Current number of elements in the buffer
}

/// a duration-based bucket for the stability metric
#[derive(Copy, Clone)]
pub struct Tau {
    value: f64,
    count: u64,
    tau: usize,
}

impl Tau {
    // construct a new `Tau`
    fn new(tau: usize) -> Tau {
        Tau {
            value: 0.0_f64,
            count: 0_u64,
            tau: tau,
        }
    }

    /// returns the time value of the `Tau`
    pub fn tau(self) -> usize {
        self.tau
    }

    // add a value to the `Tau`
    fn add(&mut self, value: f64) {
        self.value += value;
        self.count += 1;
    }

    /// returns the count of samples at `Tau`
    pub fn count(self) -> u64 {
        self.count
    }

    // return the sum at `Tau`
    pub fn value(self) -> f64 {
        self.value
    }

    /// returns the Allan Variance at `Tau`
    pub fn variance(self) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        Some(self.value() / (2.0_f64 * self.count() as f64 * (self.tau() * self.tau()) as f64))
    }

    /// returns the Allan Deviation at `Tau`
    pub fn deviation(self) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        Some(
            (self.value() / (2.0_f64 * self.count() as f64 * (self.tau() * self.tau()) as f64))
                .sqrt(),
        )
    }
}

/// describes the gaps between `Tau` and impacts space and computational costs
#[derive(Copy, Clone)]
pub enum Style {
    SingleTau(usize), // single specified Tau
    AllTau,           // all Tau from 1 ... Tau (inclusive)
    Decade,           // 1,10,100, ... Tau (inclusive)
    DecadeDeci,       // 1, 2, 3, .., 9, 10, 20, 30, .. Tau (inclusive)
    Decade124,        // 1, 2, 4, 10, 20, 40, ... Tau (inclusive)
    Decade1248,       // 1, 2, 4, 8, 10, 20, 40, ... Tau (inclusive)
    Decade125,        // 1, 2, 5, 10, 20, 50, ... Tau (inclusive)
}

/// used to configure an `Allan`
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
}

impl Default for Allan {
    fn default() -> Allan {
        Config::default().build().unwrap()
    }
}

impl Allan {
    /// create a new Allan
    pub fn new() -> Allan {
        Default::default()
    }

    fn decade_tau(max: usize, steps: Vec<usize>) -> Vec<Tau> {
        let mut p = 0;
        let mut t = 1;
        let mut taus: Vec<Tau> = Vec::new();

        while t <= max {
            for i in &steps {
                t = i * 10_u32.pow(p) as usize;
                if t <= max {
                    taus.push(Tau::new(t));
                }
            }
            p += 1;
        }
        taus
    }

    pub fn configure() -> Config {
        Config::default()
    }

    fn configured(config: Config) -> Option<Allan> {
        let samples = config.max_tau * 2 + 1; // this will vary by type

        let mut buffer = Vec::with_capacity(samples);
        // Initialize buffer with zeros
        buffer.resize(samples, 0.0);

        let mut taus: Vec<Tau> = Vec::new();

        match config.style {
            Style::SingleTau(t) => {
                taus.push(Tau::new(t));
            }
            Style::AllTau => {
                for t in 1..(config.max_tau + 1) {
                    taus.push(Tau::new(t));
                }
            }
            Style::Decade125 => taus = Allan::decade_tau(config.max_tau, vec![1, 2, 5]),
            Style::Decade124 => taus = Allan::decade_tau(config.max_tau, vec![1, 2, 4]),
            Style::Decade1248 => taus = Allan::decade_tau(config.max_tau, vec![1, 2, 4, 8]),
            Style::DecadeDeci => {
                taus = Allan::decade_tau(config.max_tau, vec![1, 2, 3, 4, 5, 6, 7, 8, 9])
            }
            Style::Decade => taus = Allan::decade_tau(config.max_tau, vec![1]),
        }

        Some(Allan {
            buffer,
            config,
            samples,
            taus,
            head: 0,
            len: 0,
        })
    }

    /// add a record
    pub fn record(&mut self, value: f64) {
        if self.len < self.samples {
            // Buffer not full yet, add to the end
            self.buffer[self.len] = value;
            self.len += 1;
        } else {
            // Buffer is full, replace oldest value
            self.buffer[self.head] = value;
            self.head = (self.head + 1) % self.samples;
        }

        // Calculate after adding the value
        self.calculate();
    }

    // recalculate values
    fn calculate(&mut self) {
        let len = self.len;
        let head = self.head;
        let samples = self.samples;

        for tau in &mut self.taus {
            let t = tau.tau() as usize;
            // Need at least 2*tau + 1 samples
            if len > 2 * t {
                // Calculate indices inline to avoid borrow issues
                let (idx0, idx1, idx2) = if len < samples {
                    // Buffer not wrapped yet
                    (len - 1, len - 1 - t, len - 1 - 2 * t)
                } else {
                    // Buffer has wrapped, use circular indexing
                    (
                        (head + samples - 1) % samples,
                        (head + samples - 1 - t) % samples,
                        (head + samples - 1 - 2 * t) % samples,
                    )
                };

                let var: f64 = self.buffer[idx2] - 2.0_f64 * self.buffer[idx1] + self.buffer[idx0];
                tau.add(var * var);
            }
        }
    }

    /// print deviations for all `Tau`
    pub fn print(&self) {
        for tau in &self.taus {
            if tau.count() >= 3 {
                println!("{} {}", tau.variance().unwrap_or(0.0), tau.tau());
            } else {
                println!("0.0 {}", tau.tau())
            }
        }
    }

    /// get a single `Tau` from the `Allan`
    pub fn get(&self, tau: usize) -> Option<Tau> {
        if tau > self.config.max_tau {
            return None;
        }
        for t in &self.taus {
            if t.tau() == tau {
                return Some(*t);
            }
        }
        None
    }
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
            if t.count() > 3 {
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
            if t.count() > 0 {
                let avar = t.variance().unwrap();
                let adev = t.deviation().unwrap();
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
            let v = allan
                .get(t)
                .unwrap_or_else(|| {
                    print!("error fetching for tau: {}", t);
                    panic!("error")
                })
                .deviation()
                .unwrap()
                * t as f64;
            if v <= 0.4 || v >= 0.6 {
                panic!("tau: {} value: {} outside of range", t, v);
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
            let v = allan
                .get(t)
                .unwrap_or_else(|| {
                    println!("error fetching for tau: {}", t);
                    panic!("error")
                })
                .deviation()
                .unwrap()
                * t as f64
                * 0.5;
            if v <= 0.1 || v >= 0.3 {
                panic!("tau: {} value: {} outside of range", t, v);
            }
        }
    }
}
