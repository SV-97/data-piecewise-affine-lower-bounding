.PHONY: all clean generate run_bench

all: generate

generate:
	pixi run generate-data

clean:
	rm -rf data
	cd generate_data && cargo clean
	cd run_benchmarks/from_rust && cargo clean
	pixi clean

run_bench:
	pixi run rs-bench experiment1 experiment2 experiment3
	pixi run py-bench experiment1 experiment2 experiment3
	pixi run R-bench experiment1 experiment2 experiment3
