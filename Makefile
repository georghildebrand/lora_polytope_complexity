.PHONY: all baseline bias depth clean

all: baseline bias depth

baseline:
	@echo "==========================================="
	@echo "Running Baseline Experiment..."
	@echo "==========================================="
	python -m experiments.run_baseline

bias:
	@echo "==========================================="
	@echo "Running Bias Ablation Study..."
	@echo "==========================================="
	python -m experiments.run_bias_ablation

depth:
	@echo "==========================================="
	@echo "Running Depth Composition Study..."
	@echo "==========================================="
	python -m experiments.run_depth

clean:
	@echo "Cleaning up results directory..."
	rm -rf results/logs/*.json
	rm -rf results/figures/*.png
