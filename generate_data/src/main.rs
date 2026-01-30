use itertools::izip;
use rand::distr::{Distribution, Uniform};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use rand_distr::num_traits::Float;
use rand_distr::{Bernoulli, Cauchy, Exp, Exp1};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use mimalloc::MiMalloc;

static N_LINES: usize = 1000;
static N_LINES_LARGE: usize = 100;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    pretty_env_logger::init();

    let data_dir = Path::new("../data");
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(20)
        .build()
        .unwrap();
    pool.install(|| {
        log::info!("Generating Experiment 1");
        experiment1(data_dir.join("experiment1"));
        log::info!("Done");
        log::info!("Generating Experiment 2");
        experiment2(data_dir.join("experiment2"));
        log::info!("Done");
        log::info!("Generating Experiment 3");
        experiment3(data_dir.join("experiment3"));
        log::info!("Done");
    });
}

#[derive(Debug, Clone, Copy)]
pub struct Laplace<T>
where
    T: Float,
    Exp1: Distribution<T>,
{
    scale: T,
    loc: T,
    exp: Exp<T>,
}

impl<T> Laplace<T>
where
    T: Float,
    Exp1: Distribution<T>,
{
    pub fn new(loc: T, scale: T) -> Result<Self, String> {
        let exp = Exp::new(T::one() / scale).map_err(|e| e.to_string())?;
        Ok(Self { scale, loc, exp })
    }
}

impl<T> Distribution<T> for Laplace<T>
where
    T: Float,
    Exp1: Distribution<T>,
{
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> T {
        let e1 = self.exp.sample(rng);
        let e2 = self.exp.sample(rng);
        self.loc + e2 - e1
    }
}

#[derive(Serialize, Deserialize)]
struct Sample {
    uid: u32,
    ground_truth: String,
    xs: Vec<f64>,
    ys: Vec<f64>,
}

/// This function is a bijection (ignoring potential overflows)
fn cantor_pairing(n: usize, m: usize) -> usize {
    ((n + m + 1) * (n + m)) / 2 + n
}

/// Generates a bunch of lines in the form (slope, intercept, uid) where the uid simply ranges from 0 to n-1.
fn lines_in_unit_square(num_lines: usize, mut rng_lines: impl Rng) -> Vec<(f64, f64, usize)> {
    let mut lines = Vec::new();
    for uid in 0..num_lines {
        let a = rng_lines.random_range(0.0..1.0);
        let b = rng_lines.random_range(0.0..1.0);
        let slope = b - a;
        let intercept = a;
        // let scale = rng_lines.random_range(0.0..100.0);
        lines.push((slope, intercept, uid));
    }
    lines
}

fn experiment1(out_dir: PathBuf) {
    let n_points_per_sample = vec![
        10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000, 200_000,
        500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000,
    ];
    let num_lines = N_LINES;
    let mut rng_lines = ChaChaRng::seed_from_u64(0);
    let uniform = Uniform::new(0.0, 1.0).unwrap();
    let laplace = Laplace::new(0.0, 0.1).unwrap();
    let small_uniform = Uniform::new(-0.05, 0.05).unwrap();

    // Generate lines in [0,1]×[0,1]
    let lines = lines_in_unit_square(num_lines, &mut rng_lines);
    log::info!(target: "exp1", "Generated Lines");

    fs::create_dir_all(&out_dir).expect("Failed to create directory");

    for n in &n_points_per_sample {
        fs::create_dir_all(out_dir.join(format!("n{n}"))).expect("Failed to create directory");
    }
    log::info!(target: "exp1", "Generated Directories");

    let line_samples = n_points_per_sample.par_iter().map(|&n| {
        let samples: Vec<Sample> = lines
            .iter()
            .filter(|(_, _, uid)| n <= 10_000 || *uid < N_LINES_LARGE)
            .map(|&(slope, intercept, uid)| {
                let mut rng = ChaChaRng::seed_from_u64(cantor_pairing(n, uid) as u64);
                let xs: Vec<f64> = uniform.sample_iter(&mut rng).take(n).collect();
                let noise: Vec<f64> = laplace.sample_iter(&mut rng).take(n).collect();
                let noise2: Vec<f64> = small_uniform.sample_iter(&mut rng).take(n).collect();
                let ys: Vec<f64> = izip!(xs.iter(), noise, noise2)
                    .map(|(&x, n, n2)| slope * x + intercept + n + n2)
                    .collect();
                let ground_truth = format!(
                    "{} * x + {} + Laplace({}, {}) + Uniform({}, {})",
                    slope, intercept, laplace.loc, laplace.scale, -0.05, 0.05,
                );
                Sample {
                    uid: u32::try_from(uid).unwrap(),
                    ground_truth,
                    xs,
                    ys,
                }
            })
            .collect();
        log::info!(target: "exp1", "Generated samples for n={n}");
        (n, samples)
    });

    line_samples
        .flat_map(|(n, samples)| samples.into_par_iter().map(move |s| (n, s)))
        .for_each(|(n, sample)| {
            let file_path = out_dir.join(format!("n{n}/{:0>10}.json", sample.uid));
            let file = File::create(file_path).expect("Failed to create file");
            let mut writer = BufWriter::new(file);

            // store xs and ys as base64 encoded binary blob
            //let xs_binary = b64.encode(bytemuck::cast_slice(&sample.xs));
            //let ys_binary = b64.encode(bytemuck::cast_slice(&sample.ys));
            //let json_data = serde_json::json!({
            //    "uid": sample.uid,
            //    "ground_truth": sample.ground_truth,
            //    "xs": xs_binary,
            //    "ys": ys_binary
            //});

            let json_data = serde_json::json!(sample);
            writeln!(writer, "{}", serde_json::to_string(&json_data).unwrap())
                .expect("Failed to write to file");
        });
}

/// Compute binomial coefficient n over k
fn comb(n: u32, k: u32) -> f64 {
    if k > n {
        0.0
    } else {
        (1..=k).map(|i| (n - k + i) as f64 / i as f64).product()
    }
}

struct Poly {
    coefficients: Vec<f64>,
}

impl Poly {
    //pub fn new(coefficients: Vec<f64>) -> Self {
    //    Poly { coefficients }
    //}

    /// Generates a random polynomial (its monomial coefficients) whose coefficients are uniformly distributed in
    /// [0,1] in the bernstein basis.
    pub fn new_bernstein_uniform<R: Rng>(rng: &mut R, dofs: usize) -> Self {
        let coeff: Vec<f64> = (0..dofs).map(|_| rng.random_range(0.0..1.0)).collect();

        // computes the matrix product `T * coeff` where T is the basis transition matrix from bernstein
        // to monomial basis
        let coefficients: Vec<f64> = (0..dofs)
            .map(|r| {
                (0..dofs)
                    .map(|k| {
                        if r >= k {
                            comb(dofs as u32 - 1, k as u32)
                                * comb(dofs as u32 - 1 - k as u32, r as u32 - k as u32)
                                * (-1.0f64).powi(r as i32 - k as i32)
                        } else {
                            0.0
                        }
                    })
                    .zip(&coeff)
                    .map(|(b_ij, c_j)| b_ij * c_j)
                    .sum::<f64>()
            })
            .collect();

        Self { coefficients }
    }

    pub fn evaluate(&self, x: f64) -> f64 {
        // Use Horner's method for efficient polynomial evaluation
        self.coefficients
            .iter()
            .rev()
            .fold(0.0, |acc, &coeff| acc * x + coeff)
    }
}

impl fmt::Display for Poly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut terms = Vec::new();

        for (i, &coeff) in self.coefficients.iter().enumerate() {
            if coeff == 0.0 {
                continue; // Skip zero coefficients
            }

            let term = match i {
                0 => format!("{:.3}", coeff),        // Constant term
                1 => format!("{:.3}x", coeff),       // Linear term
                _ => format!("{:.3}x^{}", coeff, i), // Higher-order terms
            };
            terms.push(term);
        }

        if terms.is_empty() {
            write!(f, "P(x) = 0") // Handle the case where all coefficients are zero
        } else {
            write!(f, "P(x) = {}", terms.join(" + "))
        }
    }
}

fn experiment2(out_dir: PathBuf) {
    let n_points_per_sample = vec![
        10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000, 200_000,
        500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000,
    ];
    let num_lines = N_LINES;
    let mut rng_polys = ChaChaRng::seed_from_u64(0);
    let uniform = Uniform::new(0.0, 1.0).unwrap();
    let laplace = Laplace::new(0.0, 0.1).unwrap();
    let small_uniform = Uniform::new(-0.05, 0.05).unwrap();

    // Generate polys in [0,1]×[0,1]
    let mut polys = Vec::new();
    for uid in 0..num_lines {
        let poly = Poly::new_bernstein_uniform(&mut rng_polys, 6);
        polys.push((poly, uid));
    }
    log::info!(target: "exp2", "Generated Polynomials");

    fs::create_dir_all(&out_dir).expect("Failed to create directory");

    for n in &n_points_per_sample {
        fs::create_dir_all(out_dir.join(format!("n{n}"))).expect("Failed to create directory");
    }
    log::info!(target: "exp2", "Generated Directories");

    let line_samples = n_points_per_sample.par_iter().map(|&n| {
        let samples: Vec<Sample> = polys
            .iter()
            .filter(|(_, uid)| n <= 10_000 || *uid < N_LINES_LARGE)
            .map(|(poly, uid)| {
                let mut rng = ChaChaRng::seed_from_u64(cantor_pairing(n, *uid) as u64);
                let xs: Vec<f64> = uniform.sample_iter(&mut rng).take(n).collect();
                let noise: Vec<f64> = laplace.sample_iter(&mut rng).take(n).collect();
                let noise2: Vec<f64> = small_uniform.sample_iter(&mut rng).take(n).collect();
                let ys: Vec<f64> = xs
                    .iter()
                    .zip(noise.iter())
                    .zip(noise2.iter())
                    .map(|((&x, &n), &n2)| poly.evaluate(x) + n + n2)
                    .collect();
                let ground_truth = format!(
                    "{poly} + Laplace({}, {}) + Uniform({}, {})",
                    laplace.loc, laplace.scale, -0.05, 0.05,
                );
                Sample {
                    uid: u32::try_from(*uid).unwrap(),
                    ground_truth,
                    xs,
                    ys,
                }
            })
            .collect();
        log::info!(target: "exp2", "Generated samples for n={n}");
        (n, samples)
    });

    line_samples
        .flat_map(|(n, samples)| samples.into_par_iter().map(move |s| (n, s)))
        .for_each(|(n, sample)| {
            let file_path = out_dir.join(format!("n{n}/{:0>10}.json", sample.uid));
            let file = File::create(file_path).expect("Failed to create file");
            let mut writer = BufWriter::new(file);

            // store xs and ys as base64 encoded binary blob
            //let xs_binary = b64.encode(bytemuck::cast_slice(&sample.xs));
            //let ys_binary = b64.encode(bytemuck::cast_slice(&sample.ys));
            //let json_data = serde_json::json!({
            //    "uid": sample.uid,
            //    "ground_truth": sample.ground_truth,
            //    "xs": xs_binary,
            //    "ys": ys_binary
            //});

            let json_data = serde_json::json!(sample);
            writeln!(writer, "{}", serde_json::to_string(&json_data).unwrap())
                .expect("Failed to write to file");
        });
}

fn experiment3(out_dir: PathBuf) {
    let n_points_per_sample = vec![
        10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000, 200_000,
        500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000,
    ];
    let num_lines = N_LINES;
    let mut rng_lines = ChaChaRng::seed_from_u64(0);
    let uniform = Uniform::new(0.0, 1.0).unwrap();
    // let gauss = Normal::new(0.0, 0.1).unwrap();
    let laplace = Laplace::new(0.0, 0.01).unwrap();
    let cauchy = Cauchy::new(0.0, 0.5).unwrap();
    let bernoulli = Bernoulli::new(0.05).unwrap(); // true indicates outlier

    // Generate lines in [0,1]×[0,1]
    let lines = lines_in_unit_square(num_lines, &mut rng_lines);
    log::info!(target: "exp3", "Generated Lines");

    fs::create_dir_all(&out_dir).expect("Failed to create directory");

    for n in &n_points_per_sample {
        fs::create_dir_all(out_dir.join(format!("n{n}"))).expect("Failed to create directory");
    }
    log::info!(target: "exp3", "Generated Directories");

    let line_samples = n_points_per_sample.par_iter().map(|&n| {
        let samples: Vec<Sample> = lines
            .iter()
            .filter(|(_, _, uid)| n <= 10_000 || *uid < N_LINES_LARGE)
            .map(|&(slope, intercept, uid)| {
                let mut rng = ChaChaRng::seed_from_u64(cantor_pairing(n, uid) as u64);
                let xs: Vec<f64> = uniform.sample_iter(&mut rng).take(n).collect();
                let laplace_noise: Vec<f64> = laplace.sample_iter(&mut rng).take(n).collect();
                let cauchy_noise: Vec<f64> = cauchy.sample_iter(&mut rng).take(n).collect();
                let is_outlier: Vec<bool> = bernoulli.sample_iter(&mut rng).take(n).collect();
                let ys: Vec<f64> = izip!(xs.iter(), laplace_noise, cauchy_noise, is_outlier)
                    .map(|(x, ln, cn, is_outlier)| {
                        slope * x + intercept + (if is_outlier { cn } else { ln })
                    })
                    .collect();

                let ground_truth = format!(
                    "{} * x + {} + Mixture(Laplace(0,{}), Cauchy(0,{}), {})",
                    slope, intercept, 0.01, 0.5, 0.05
                );
                Sample {
                    uid: u32::try_from(uid).unwrap(),
                    ground_truth,
                    xs,
                    ys,
                }
            })
            .collect();
        log::info!(target: "exp3", "Generated samples for n={n}");
        (n, samples)
    });

    line_samples
        .flat_map(|(n, samples)| samples.into_par_iter().map(move |s| (n, s)))
        .for_each(|(n, sample)| {
            let file_path = out_dir.join(format!("n{n}/{:0>10}.json", sample.uid));
            let file = File::create(file_path).expect("Failed to create file");
            let mut writer = BufWriter::new(file);

            // store xs and ys as base64 encoded binary blob
            //let xs_binary = b64.encode(bytemuck::cast_slice(&sample.xs));
            //let ys_binary = b64.encode(bytemuck::cast_slice(&sample.ys));
            //let json_data = serde_json::json!({
            //    "uid": sample.uid,
            //    "ground_truth": sample.ground_truth,
            //    "xs": xs_binary,
            //    "ys": ys_binary
            //});

            let json_data = serde_json::json!(sample);
            writeln!(writer, "{}", serde_json::to_string(&json_data).unwrap())
                .expect("Failed to write to file");
        });
}
