//! Allan provides variance and deviation tools for stability analysis
//!
//! # Goals
//! * provide streaming variance and deviations from series data
//! * pre-allocated datastructures
//!
//! # Future work
//! * actually finishing it
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
//! let mut adev = Allan::new().unwrap();

#![crate_type = "lib"]

use std::collections::VecDeque;
extern crate rand;

pub struct Allan {
    buffer: VecDeque<f64>,
    samples: u32,
    taus: Vec<Tau>,
}

#[derive(Copy, Clone)]
struct Tau {
    value: f64,
    count: u32,
    tau: u32,
}

impl Tau {
    pub fn new(tau: u32) -> Tau {
        Tau {
            value: 0.0_f64,
            count: 0_u32,
            tau: tau,
        }
    }

    pub fn tau(self) -> u32 {
        self.tau
    }

    pub fn add(&mut self, value: f64) {
        self.value += value;
        self.count += 1;
    }

    pub fn count(self) -> u32 {
        self.count
    }

    pub fn value(self) -> f64 {
        self.value
    }
}

pub enum AllanStyle {
    SingleTau(u64), // single specified Tau
    AllTau, // all Tau from 1 ... Tau (inclusive)
    Decade, // 1,10,100, ... Tau (inclusive)
    DecadeDeci, // 1, 2, 3, .., 9, 10, 20, 30, .. Tau (inclusive)
    Decade124, // 1, 2, 4, 10, 20, 40, ... Tau (inclusive)
    Decade1248, // 1, 2, 4, 8, 10, 20, 40, ... Tau (inclusive)
    Decade125, // 1, 2, 5, 10, 20, 50, ... Tau (inclusive)
    Octave, // 1, 2, 4, 8, 16, 32, ... Tau (inclusive)
}

pub struct AllanConfig {
    max_tau: usize,
    style: AllanStyle,
}

impl AllanConfig {
    pub fn new() -> AllanConfig {
        Default::default()
    }

    pub fn style(&mut self, style: AllanStyle) -> &Self {
        self.style = style;
        self
    }

    pub fn max_tau(&mut self, max_tau: usize) -> &Self {
        self.max_tau = max_tau;
        self
    }
}

impl Default for AllanConfig {
    fn default() -> AllanConfig {
        AllanConfig {
            max_tau: 1_000,
            style: AllanStyle::DecadeDeci,
        }
    }
}

impl Allan {
    /// create a new Allan
    pub fn new() -> Option<Allan> {
        let config = AllanConfig::new();
        Allan::configured(config)
    }

    fn decade_tau(max: u32, steps: Vec<u32>) -> Vec<Tau> {
        let mut p = 0;
        let mut t = 1_u32;
        let mut taus: Vec<Tau> = Vec::new();

        while t <= max {
            for i in &steps {
                t = i * 10_u32.pow(p);
                if t <= max {
                    taus.push(Tau::new(t));
                }
            }
            p += 1;
        }
        taus
    }

    pub fn configured(config: AllanConfig) -> Option<Allan> {
        let samples: u32 = (config.max_tau * 2 + 1) as u32; // this will vary by type

        let buffer = VecDeque::with_capacity(samples as usize);

        let mut taus: Vec<Tau> = Vec::new();

        match config.style {
            AllanStyle::AllTau => {
                for t in 1..(config.max_tau + 1) {
                    taus.push(Tau::new(t as u32));
                }
            }
            AllanStyle::Decade125 => taus = Allan::decade_tau(config.max_tau as u32, vec![1, 2, 5]),
            AllanStyle::Decade124 => taus = Allan::decade_tau(config.max_tau as u32, vec![1, 2, 4]),
            AllanStyle::Decade1248 => {
                taus = Allan::decade_tau(config.max_tau as u32, vec![1, 2, 4, 8])
            }
            AllanStyle::DecadeDeci => {
                taus = Allan::decade_tau(config.max_tau as u32, vec![1, 2, 3, 4, 5, 6, 7, 8, 9])
            }
            AllanStyle::Decade => taus = Allan::decade_tau(config.max_tau as u32, vec![1]),
            _ => {}
        }

        Some(Allan {
            buffer: buffer,
            samples: samples,
            taus: taus,
        })
    }

    /// add a record
    pub fn record(&mut self, value: f64) {
        self.buffer.push_front(value);
        self.calculate();
        if self.buffer.len() as u32 == self.samples {
            let _ = self.buffer.pop_back();
        }
    }

    /// recalculate values
    pub fn calculate(&mut self) {
        for tau in &mut self.taus {
            let t = tau.tau() as usize;
            if (2 * t) < self.buffer.len() {
                // println!("calculating for: {}", t);
                let var: f64 = self.buffer[(2 * t)] - 2.0_f64 * self.buffer[t] + self.buffer[0];
                tau.add(var.powf(2.0_f64));
            } else {
                // println!("tau: {} too large", t);
            }
        }
    }

    // print things out
    pub fn print(&mut self) {
        for tau in &mut self.taus {
            if tau.count() >= 3 {
                let dev = tau.value() / (2.0_f64 * tau.count() as f64);
                let dev = dev.powf(0.5_f64);
                let dev = dev / tau.tau() as f64;
                println!("{} {}", dev, tau.tau());
            } else {
                println!("0.0 {}", tau.tau())
            }
        }
    }
}
