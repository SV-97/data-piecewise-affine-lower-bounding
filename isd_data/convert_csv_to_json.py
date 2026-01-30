# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars",
# ]
# ///
import json
from collections import defaultdict
import datetime as dt
from pathlib import Path
import polars as pl

def convert_csvs_to_json(aggregated_out_dir: Path, out_dir: Path, noisy: bool = False):
    """
    Convert CSV files in aggregated_out_dir to JSON format for regression consumption.
    Datetimes are scaled so that year 1950.0 corresponds to xs = 0.0,
    and each year corresponds to 1.0 units.
    """
    # Precompute scaling constants
    ref_dt = dt.datetime(1950, 1, 1, tzinfo=dt.timezone.utc)
    ref_ts = ref_dt.timestamp()
    seconds_per_year = 365.25 * 24 * 3600

    folder_counters = defaultdict(int)  # Per row-count numbering
    global_uid = 0

    for file in sorted(aggregated_out_dir.iterdir()):
        if file.suffix.lower() != ".csv":
            continue

        if noisy:
            print(f"Reading: {file.name}")

        # Read CSV and drop rows with nulls
        df = pl.read_csv(
            file,
            schema={"datetime_utc": pl.Datetime, "dry_bulb_c": pl.Float64}
        ).drop_nulls()

        if df.is_empty():
            if noisy:
                print(f"  Skipped (empty after dropping nulls)")
            continue
        if len(df) < 10:
            if noisy:
                print(f"  Skipped (too short after dropping nulls)")
            continue

        row_count = len(df)

        # Convert datetime to scaled year offsets from 1950
        timestamps = df["datetime_utc"].dt.timestamp().to_list()
        xs = [(ts - ref_ts) / seconds_per_year for ts in timestamps]

        # Extract dry bulb temperatures
        ys = df["dry_bulb_c"].to_list()

        # Prepare JSON structure
        data = {
            "ground_truth": file.name,
            "uid": global_uid,
            "xs": xs,
            "ys": ys
        }

        # Output directory for this row count
        row_count_dir = out_dir / f"n{row_count}"
        row_count_dir.mkdir(parents=True, exist_ok=True)

        # Numbered filename inside folder
        file_number = folder_counters[row_count]
        out_file_name = f"{file_number:010d}.json"
        out_file_path = row_count_dir / out_file_name

        # Write JSON
        with open(out_file_path, "w") as f:
            json.dump(data, f)

        if noisy:
            print(f"  Rows: {row_count}")
            print(f"  UID: {global_uid}")
            print(f"  Local number: {file_number}")
            print(f"  Output: {out_file_path}")

        # Increment counters
        folder_counters[row_count] += 1
        global_uid += 1


aggregated_out_dir = Path("./aggregated_by_station")
convert_csvs_to_json(aggregated_out_dir, Path("../data/experiment4"), True)
