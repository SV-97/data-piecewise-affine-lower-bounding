# Data reproduction for *Fast and Exact Least Absolute Deviations Line Fitting via Piecewise Affine Lower-Bounding*

This data contains all files needed to reproduce the experiments from the paper *Fast and Exact Least Absolute Deviations Line Fitting via Piecewise Affine Lower-Bounding*. For further background information please see the main repository [https://github.com/SV-97/piecewise-affine-lower-bounding](https://github.com/SV-97/piecewise-affine-lower-bounding).

## Setting up the Environment

All dependencies and necessary software except for the proprietary CPLEX can be installed automatically via Pixi, a cross-platform package manager.
See the installation instructions for pixi here [https://pixi.prefix.dev/latest/installation/](https://pixi.prefix.dev/latest/installation/) or check out the project's page here [https://prefix.dev/](https://prefix.dev/).

With this (and CPLEX, depending on what you want to do) installed you can directly run the commands from this readme and it'll automatically download the necessary dependencies to a dedicated environment and set everything up. It should not impact your system configuration in any way.

We also include a makefile that allows you to generate the data, run all benchmarks and clean up the environment at the end. For this reason make is installed into the pixi environment, you can invoke it like this:
```console
pixi run make generate
pixi run make clean
```

All pixi commands can be called from anywhere within the project directory.

If you wish to not use pixi you can find all dependencies and shell commands in the [Makefile](Makefile) and [pixi.toml](pixi.toml) files (check the `[tasks]` section).

## Experiment input data

### Generating synthetic data

To generate the synthetic data please execute the program in [generate_data](generate_data) either manually (c.f. the makefile) or by running
```console
pixi run generate-data
```

This generates a substantial amount of json data and as such takes a while to execute.
If you only want to generate parts of the data you can do so by commenting out the other experiments in [generate_data/src/main.rs](generate_data/src/main.rs).

### Preparing Real-world Data

The identifiers of all stations whose data we downloaded from the ISD database are given in the file [isd_data/isd_stations.txt](isd_data/isd_stations.txt) in the format `{usaf}-{wban}`, one per line. You can download the data from the [NOAA ISD database](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database).
Post-download the ISD data has to be aggregated per station into a csv file matching the schema given in [isd_data/aggregated_by_station/010013-99999.csv](isd_data/aggregated_by_station/010013-99999.csv),
and from there converted into the actual input files for the benchmark runners.
This latter step can be accomplished via the provided python script [isd_data/make_experiment_files.py](isd_data/make_experiment_files.py). Run this script via
```console
pixi run isd-json-from-csv
```
This should generate an `experiment4` directory in `./data` for later consumption by the benchmark runners.

## Running numerical experiments

To run experiments use the following commands:
```console
pixi run rs-bench experiment1 experiment2 experiment3 experiment4
pixi run py-bench experiment1 experiment2 experiment3 experiment4
pixi run R-bench experiment1 experiment2 experiment3 experiment4
```
or use the makefile to consecutively run experiments 1 through 3 via
```console
pixi run make run_bench
```
Please read the note regarding experiment 4 at the bottom!

Note that running all experiments should be expected to take a significant amount of time to complete (on the order of hours).
To run just a part of the experiments use the above commands but only supply the experiments you want to run, for example:
```console
pixi run rs-bench experiment1 experiment2
```
to run just the first two experiments with all rust-based methods.

> [!WARNING]
> Running CPLEX experiments requires a local CPLEX installation and license.
> A brief explanation on how to get a license at the time of writing is given in [how_to_setup_cplex.md](how_to_setup_cplex.md).
> Further you have to set the environment variable `CPLEX_PATH` to the path of your CPLEX installation.
> The [pixi.toml](pixi.toml) file contains a standard path that may work for you, but you may need to adjust it depending on your system configuration.

> [!CAUTION]
> Be aware that experiment 4 is not special-cased in the code and requires some manual changes to the benchmark code.
> In particular the size bounds have to be disabled for all methods (e.g. by commenting them out).
> For this reason, experiment 4 is also not included in the default makefile target.
