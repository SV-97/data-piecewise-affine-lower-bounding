import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import statsmodels.api as sm
from pandas.io.formats.style_render import Optional
from statsmodels.regression.quantile_regression import QuantReg
from tqdm import tqdm


@dataclass
class LADResult:
    """A simple dataclass to store the results of a Least Absolute Deviations (LAD) regression."""

    slope: float
    intercept: float
    sum_abs_deviations: float


# uses LP with HiGHS as solver
from sklearn.linear_model import QuantileRegressor


def time_sklearn(xs, ys) -> float:
    start = time.perf_counter()
    least_abs_line_sklearn(xs, ys)
    time_taken = time.perf_counter() - start
    return time_taken


def least_abs_line_sklearn(xs: np.ndarray, ys: np.ndarray) -> LADResult:
    """
    Solve the Least Absolute Deviations (LAD) regression problem using scikit-learn's QuantileRegressor.

    Args:
        xs (np.ndarray): Array of x-coordinates.
        ys (np.ndarray): Array of y-coordinates.

    Returns:
        LADResult: An object containing the slope (m), intercept (t), and the
        sum of absolute deviations.
    """
    N = len(xs)
    if len(ys) != N:
        raise ValueError("Input arrays xs and ys must have the same length.")

    X = xs.reshape(-1, 1)
    y = ys

    quant_reg = QuantileRegressor(quantile=0.5, solver="highs")
    quant_reg.fit(X, y)

    # Extract the slope (coefficient) and intercept from the fitted model
    slope = quant_reg.coef_[0]
    intercept = quant_reg.intercept_

    predictions = quant_reg.predict(X)

    sum_abs_deviations = np.sum(np.abs(y - predictions))

    return LADResult(
        slope=slope, intercept=intercept, sum_abs_deviations=sum_abs_deviations
    )


def time_statsmodels(xs, ys) -> tuple[float, Optional[float]]:
    start = time.perf_counter()
    res = least_abs_line_statsmodels(xs, ys)
    time_taken = time.perf_counter() - start
    if res is None:  # method did not terminate
        return float("inf"), float("NaN")
    else:
        return time_taken, res.sum_abs_deviations


def least_abs_line_statsmodels(xs: np.ndarray, ys: np.ndarray) -> Optional[LADResult]:
    """
    Solve the Least Absolute Deviations (LAD) regression problem using statsmodels' QuantReg.

    Args:
        xs (np.ndarray): Array of x-coordinates.
        ys (np.ndarray): Array of y-coordinates.

    Returns:
        Optional[LADResult]: An object containing the slope (m), intercept (t), and the
        sum of absolute deviations. Returns None if the method does not converge.
    """
    N = len(xs)
    if len(ys) != N:
        raise ValueError("Input arrays xs and ys must have the same length.")

    X = sm.add_constant(xs)
    y = ys

    quant_reg = QuantReg(y, X)

    try:
        # We can change the warning into an exception to catch it
        with warnings.catch_warnings():
            warnings.simplefilter("error", sm.tools.sm_exceptions.IterationLimitWarning)
            results = quant_reg.fit(q=0.5)

            # Extract the slope and intercept from the results object
            # The first coefficient is the intercept (from the constant we added)
            # and the second is the slope
            slope = results.params[1]
            intercept = results.params[0]

            # Calculate the sum of absolute deviations from the residuals
            sum_abs_deviations = np.sum(np.abs(results.resid))

            return LADResult(
                slope=slope, intercept=intercept, sum_abs_deviations=sum_abs_deviations
            )
    except sm.tools.sm_exceptions.IterationLimitWarning:
        # Return None on failure: the method did not converge so there is no solution
        return None


def process_json_files(experiment_directory: Path, out_dir: Path):
    os.makedirs(out_dir, exist_ok=True)

    file_paths = [
        p
        for sample_folder in experiment_directory.iterdir()
        if sample_folder.is_dir()
        for p in sample_folder.iterdir()
        if p.suffix == ".json"
    ]

    # Shuffle file paths for better progress tracking
    import random

    random.shuffle(file_paths)

    number_of_methods = 3
    total_steps = len(file_paths) * number_of_methods
    progress_bar = tqdm(total=total_steps, desc="Processing files", unit="step")

    times_rows = []
    for file_path in file_paths:
        n = int(file_path.parent.name.lstrip("n"))
        if n > 2_000_000:
            times_rows.append((n, None, None, None, None, None))
            progress_bar.update(number_of_methods)
            continue

        with open(file_path, "r") as f:
            sample = json.load(f)

        uid = sample.get("uid")
        xs, ys = sample.get("xs", []), sample.get("ys", [])
        xs = np.array(xs)
        ys = np.array(ys)
        assert len(xs) == len(ys), "Mismatch in xs and ys length"

        n = len(xs)
        # points = list(zip(xs, ys))
        cplex_time = None  # time_cplex(xs, ys)
        progress_bar.update(1)
        sklearn_time = None  # time_sklearn(xs,ys) if n <= 5000 else None
        progress_bar.update(1)
        statsmodels_time, sm_obj = time_statsmodels(xs, ys)
        progress_bar.update(1)

        times_rows.append((n, uid, cplex_time, sklearn_time, statsmodels_time, sm_obj))

    progress_bar.close()

    # Construct DataFrame
    df = pl.DataFrame(
        times_rows,
        schema=[
            "n_samples",
            "uid",
            "CPLEX",
            "scikit-learn",
            "statsmodels",
            "statsmodels (obj)",
        ],
        orient="row",
    )

    # Write to CSV
    output_file = out_dir / f"{experiment_directory.name}_python.csv"
    df.write_csv(output_file)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    data_directory = Path("../../data")
    output_directory = Path("../../out")
    for arg in sys.argv[1:]:
        print(f"Processing experiment {arg}")
        experiment_directory = data_directory / arg  # "experiment1" etc.
        process_json_files(experiment_directory, output_directory)
