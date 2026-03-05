.PHONY: all baseline bias depth rank_sweep random_control input_dim clean

all: baseline bias depth rank_sweep random_control input_dim

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

rank_sweep:
	@echo "==========================================="
	@echo "Running Rank Sweep Experiment..."
	@echo "==========================================="
	python -m experiments.run_rank_sweep

random_control:
	@echo "==========================================="
	@echo "Running Random Low-Rank Control..."
	@echo "==========================================="
	python -m experiments.run_random_lowrank_control

input_dim:
	@echo "==========================================="
	@echo "Running Ambient Input Dimension Ablation..."
	@echo "==========================================="
	python -m experiments.run_input_dim_ablation

clean:
	@echo "Cleaning up results directory..."
	rm -rf results/logs/*.json
	rm -rf results/figures/*.png
