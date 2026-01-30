use std::{
    env, fs,
    hint::black_box,
    path::Path,
    time::{Duration, Instant},
};

use good_lp::{
    DualValues, Expression, Solution, SolutionWithDual, Solver, SolverModel, constraint,
    solvers::{clarabel, cplex, highs},
    variable::UnsolvedProblem,
    variables,
};

use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use ordered_float::OrderedFloat;
use palb::{Floating, PrimalLine, PrimalPoint, SolverInfo, l1line_with_info, objective_value};
use polars::{
    frame::{DataFrame, row::Row},
    io::SerWriter,
    prelude::{AnyValue, CsvWriter, DataType, Schema},
};
use rand::distr::Distribution;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use rand_distr::num_traits::Float;
use rand_distr::{Exp, Exp1}; // Normal
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct Sample {
    uid: u32,
    ground_truth: String,
    xs: Vec<f64>,
    ys: Vec<f64>,
}

pub fn least_abs_line_linprog<S>(points: &[PrimalPoint], solver: S) -> Option<PrimalLine>
where
    S: Solver,
{
    let n = points.len();
    if n == 0 {
        return None;
    }

    // Define variables
    variables! {
        vars:
            m;
            t;
            z[n];
    }

    // Objective: minimize the sum of z
    let objective: UnsolvedProblem = vars.minimise((0..n).map(|i| z[i]).sum::<Expression>());

    // Add constraints for each point
    let mut problem = objective.using(solver);
    // some solvers -- notably Cbc -- like to spam info to stdout.
    // This is annoying and slows everything down needlessly.
    // Setting the loglevel to 0 disables most if not all output
    // per https://github.com/rust-or/good_lp/issues/30
    // problem.set_parameter("loglevel", "0");
    for (i, point) in points.iter().enumerate() {
        let xi = point.x();
        let yi = point.y();
        problem = problem
            .with(constraint!(
                xi.into_inner() * m + t - yi.into_inner() <= z[i]
            ))
            .with(constraint!(
                xi.into_inner() * m + t - yi.into_inner() >= -z[i]
            ));
    }

    // Solve the problem
    let solution = problem.solve().ok()?;

    // Extract the results
    let slope = solution.value(m);
    let intercept = solution.value(t);

    Some(PrimalLine {
        coords: (Floating::from(slope), Floating::from(intercept)),
    })
}

pub fn least_abs_line_linprog_dual_with_sol<S>(
    points: &[PrimalPoint],
    solver: S,
) -> Option<(PrimalLine, f64)>
where
    S: Solver,
    <<S as Solver>::Model as SolverModel>::Solution: for<'a> SolutionWithDual<'a>,
{
    let n = points.len();
    if n == 0 {
        return None;
    }

    // Define variables
    variables! {
        vars:
            -1 <= d[n] <= 1;
    }

    // Objective: minimize the sum of z
    let obj_expr: Expression = points
        .iter()
        .enumerate()
        .map(|(i, p)| p.y().into_inner() * d[i])
        .sum();
    let objective: UnsolvedProblem = vars.maximise(&obj_expr);

    // Add constraints
    let mut problem = objective.using(solver);

    let xs_dot_ds = points
        .iter()
        .enumerate()
        .map(|(i, p)| p.x().into_inner() * d[i])
        .sum::<Expression>();
    let slope_constraint = problem.add_constraint(constraint!(xs_dot_ds == 0));
    let intercept_constraint =
        problem.add_constraint(constraint!(d.iter().sum::<Expression>() == 0));

    // Solve the problem
    let mut solution = problem.solve().ok()?;
    let dual_solution = solution.compute_dual();

    let slope = dual_solution.dual(slope_constraint);
    let intercept = dual_solution.dual(intercept_constraint);

    drop(dual_solution);
    // Evaluate the original objective expression using the solution
    let optimal_objective_value = solution.eval(&obj_expr);

    Some((
        PrimalLine {
            coords: (Floating::from(slope), Floating::from(intercept)),
        },
        optimal_objective_value,
    ))
}

pub fn least_abs_line_linprog_dual<S>(points: &[PrimalPoint], solver: S) -> Option<PrimalLine>
where
    S: Solver,
{
    let n = points.len();
    if n == 0 {
        return None;
    }

    // Define variables
    variables! {
        vars:
            -1 <= d[n] <= 1;
    }

    // Objective: minimize the sum of z
    let objective: UnsolvedProblem = vars.maximise(
        points
            .iter()
            .enumerate()
            .map(|(i, p)| p.y().into_inner() * d[i])
            .sum::<Expression>(),
    );

    let mut problem = objective.using(solver);

    let xs_dot_ds = points
        .iter()
        .enumerate()
        .map(|(i, p)| p.x().into_inner() * d[i])
        .sum::<Expression>();
    let _intercept_constraint = problem.add_constraint(constraint!(xs_dot_ds == 0));
    let _slope_constraint = problem.add_constraint(constraint!(d.iter().sum::<Expression>() == 0));

    let solution = problem.solve().ok()?;

    let res = solution.value(d[0]);
    Some(PrimalLine {
        coords: (Floating::from(res), Floating::from(res)),
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct LinProgRes {
    time: Duration,
    obj_val: Option<f64>,
}

#[derive(Debug, PartialEq, Clone, Copy)]
struct LinProgResults {
    clarabel: Option<LinProgRes>,
    highs: Option<LinProgRes>,
    cplex: Option<LinProgRes>,
}

fn time_linprogs(points: &[PrimalPoint], bar: &ProgressBar) -> LinProgResults {
    if points.len() > 500_000 {
        bar.inc(3);
        return LinProgResults {
            clarabel: None,
            highs: None,
            cplex: None,
        };
    }

    let clarabel_res = if points.len() <= 200_000 {
        bar.set_message(format!("Clarabel on n={}", points.len()));
        let t2 = Instant::now();
        let sol = black_box(least_abs_line_linprog_dual_with_sol(
            points,
            clarabel::clarabel,
        ));
        if sol.is_none() {
            println!("Clarabel failed on a sample of length {}", points.len());
            //dbg!(points.len());
            //dbg!(&points);
        }
        Some(LinProgRes {
            time: t2.elapsed(),
            obj_val: sol.map(|s| *objective_value(s.0, points)),
        })
    } else {
        None
    };
    bar.inc(1);

    bar.set_message(format!("CPLEX on n={}", points.len()));
    let t3 = Instant::now();
    let _sol = black_box(least_abs_line_linprog_dual(points, cplex::cplex)).unwrap();
    let cplex_res = Some(LinProgRes {
        time: t3.elapsed(),
        obj_val: None,
    });
    bar.inc(1);

    let highs_duration = if points.len() <= 100_000 {
        bar.set_message(format!("HiGHS on n={}", points.len()));
        let t1 = Instant::now();
        let sol = black_box(least_abs_line_linprog_dual_with_sol(points, highs::highs)).unwrap();
        Some(LinProgRes {
            time: t1.elapsed(),
            obj_val: Some(*objective_value(sol.0, points)),
        })
    } else {
        None
    };
    bar.inc(1);

    LinProgResults {
        clarabel: clarabel_res,
        highs: highs_duration,
        cplex: cplex_res,
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct OurTimes {
    duration: Duration,
    obj_value: Floating,
    info: SolverInfo,
}

fn time_ours(points: &[PrimalPoint], bar: &ProgressBar) -> OurTimes {
    bar.set_message(format!("Ours on n={}", points.len()));
    let t1 = Instant::now();
    let sol = black_box(l1line_with_info::<true>(points)).unwrap();
    let duration = t1.elapsed();
    bar.inc(1);

    OurTimes {
        duration,
        obj_value: objective_value(sol.optimal_line, points),
        info: sol.info,
    }
}

pub fn find_max_input_size(
    mut initial_size: usize,
    time_limit: Duration,
    input_absolute_acc: usize,
    runtime_of_algo: impl Fn(usize) -> Duration,
) -> usize {
    // Exponential search phase
    while runtime_of_algo(initial_size) <= time_limit {
        initial_size *= 2;
    }

    // Binary search phase
    let mut low = initial_size / 2;
    let mut high = initial_size;
    while high - low > input_absolute_acc {
        let mid = (low + high) / 2;
        if runtime_of_algo(mid) <= time_limit {
            low = mid;
        } else {
            high = mid;
        }
    }

    (low + high) / 2 // Return the approximated input size that of what should runs in ≤1s
}

#[derive(Debug, Clone, Copy)]
pub struct Laplace<T>
where
    T: Float,
    Exp1: Distribution<T>,
{
    #[allow(unused)]
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

fn main() {
    let args: Vec<String> = env::args().collect();
    pretty_env_logger::init();

    let data_dir = Path::new("../../data");
    let out_dir = Path::new("../../out");
    fs::create_dir_all(out_dir).expect("Failed to create output directory.");

    let benchmark_experiments = args[1..].iter().map(|subdir| data_dir.join(subdir));

    benchmark_experiments
        .into_iter()
        .for_each(|experiment_directory| {
            log::info!("Processing directory {}", experiment_directory.as_os_str().to_string_lossy());
            let mut file_paths = fs::read_dir(&experiment_directory)
                .expect("Failed to access experiment directory")
                .flat_map(|sample_folder| fs::read_dir(sample_folder.unwrap().path()).unwrap())
                .collect_vec();
            // shuffle file paths so that progress bar is more accurate
            let mut rng = StdRng::from_os_rng();
            // file_paths.sort();
            file_paths.shuffle(&mut rng);

            let number_of_methods = 4;
            let progress_bar = ProgressBar::new(number_of_methods * file_paths.len() as u64);
            progress_bar.set_style(
                ProgressStyle::with_template(
                    "[{elapsed_precise}] [ETA: {eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
                )
                .unwrap(),
            );
            // inc dec initializes bar
            progress_bar.inc(1);
            progress_bar.dec(1);

            let times_rows = file_paths
                .into_iter()
                .filter_map(|sample| -> Option<Sample> {
                    let path = sample.unwrap().path();
                    let content = fs::read(&path).unwrap();
                    let sample = serde_json::from_slice(&content);
                    match sample {
                        Err(e) => {
                            log::error!("Error with {}: {}", path.display(), &e);
                            None
                        }
                        Ok(s) => Some(s)
                    }
                })
                .map(|sample| {
                    // println!("{}", sample.ground_truth);
                    // println!("{:?}", sample.xs);
                    // println!("{:?}", sample.ys);
                    let n = sample.xs.len();
                    if true { // n < 10_000
                    assert_eq!(sample.xs.len(), sample.ys.len());
                    let points = sample
                        .xs
                        .into_iter()
                        .zip(sample.ys)
                        .map(|(x, y)| PrimalPoint {
                            coords: (OrderedFloat::from(x), OrderedFloat::from(y)),
                        })
                        .collect::<Vec<_>>();
                    let linprog_times = time_linprogs(&points, &progress_bar);
                    let our_time = time_ours(&points, &progress_bar);
                    let row_v = vec![
                        AnyValue::UInt32(sample.uid),
                        AnyValue::UInt64(n as u64),
                        AnyValue::Float64(our_time.duration.as_secs_f64()),
                        AnyValue::from(linprog_times.clarabel.map(|res| res.time.as_secs_f64())),
                        AnyValue::from(linprog_times.cplex.map(|res| res.time.as_secs_f64())),
                        AnyValue::from(linprog_times.highs.map(|res| res.time.as_secs_f64())),
                        AnyValue::from(linprog_times.clarabel.and_then(|res| res.obj_val)),
                        AnyValue::from(linprog_times.cplex.and_then(|res| res.obj_val)),
                        AnyValue::from(linprog_times.highs.and_then(|res| res.obj_val)),
                        AnyValue::Float64(our_time.obj_value.0),
                        AnyValue::UInt64(our_time.info.num_iters as u64),
                        AnyValue::UInt64(our_time.info.num_expansion as u64),
                        AnyValue::UInt64(our_time.info.num_subdiv as u64),
                    ];
                    Row::new(row_v)
                    } else {
                        Row::new(vec![
                            AnyValue::UInt32(sample.uid),
                            AnyValue::UInt64(n as u64),
                            AnyValue::Float64(0.0),
                            AnyValue::from(0.0),
                            AnyValue::from(0.0),
                            AnyValue::from(0.0),
                            AnyValue::from(0.0),
                            AnyValue::from(0.0),
                            AnyValue::from(0.0),
                            AnyValue::Float64(0.0),
                            AnyValue::UInt64(2),
                            AnyValue::UInt64(1),
                            AnyValue::UInt64(1),
                        ])
                    }
                })
                .collect_vec();
            progress_bar.finish();

            // construct dataframe for output
            let mut schema = Schema::default();
            schema.insert("uid".into(), DataType::UInt32);
            schema.insert("n_samples".into(), DataType::UInt64);
            schema.insert("Ours".into(), DataType::Float64);
            schema.insert("Clarabel".into(), DataType::Float64);
            schema.insert("CPLEX".into(), DataType::Float64);
            schema.insert("HiGHS".into(), DataType::Float64);
            schema.insert("Clarabel (obj)".into(), DataType::Float64);
            schema.insert("CPLEX (obj)".into(), DataType::Float64);
            schema.insert("HiGHS (obj)".into(), DataType::Float64);
            schema.insert("Ours (obj)".into(), DataType::Float64);
            schema.insert("Ours (iters)".into(), DataType::UInt64);
            schema.insert("Ours (expand_iters)".into(), DataType::UInt64);
            schema.insert("Ours (subdiv_iters)".into(), DataType::UInt64);
            let mut times_df = DataFrame::from_rows_iter_and_schema(times_rows.iter(), &schema).unwrap();

            // write the dataframe out as csv
            let mut out_file = fs::File::create(out_dir.join(format!("{}_rust.csv", experiment_directory.file_name().unwrap().to_str().unwrap()))).expect("could not create file");
            CsvWriter::new(&mut out_file)
                .include_header(true)
                .with_separator(b',')
                .finish(&mut times_df).expect("Failed to write csv");
            println!(
                "Wrote {}_rust.csv", experiment_directory.file_name().unwrap().to_str().unwrap()
            );
        });
}
