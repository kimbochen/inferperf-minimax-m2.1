# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains LLM inference benchmarking tools for MiniMax M2.1 model on vLLM, including:
- **bench_serving**: Python client for running serving benchmarks against various inference backends
- **plot_results.py**: Visualization script for benchmark results
- **results/**: Directory containing benchmark result JSON files

## Repository Structure

```
inferperf_minimax-m2.1/
├── bench_serving/                    # Benchmark client (forked from vLLM benchmarks)
│   ├── benchmark_serving.py          # Main benchmark runner
│   ├── backend_request_func.py       # Backend-specific request functions
│   └── benchmark_utils.py            # Output format utilities
├── results/                          # Benchmark result JSON files
│   └── minimax_m2.1_vllm_isl{ISL}_osl{OSL}_conc{CONC}_tp{TP}.json
├── plot_results.py                   # Visualization script
├── benchmark_plot.png                # Generated plot output
└── minimax_m2.1_vllm_tp4_sbatch.sh   # Slurm batch script for running benchmarks
```

## Running Benchmarks

### Using bench_serving directly

```bash
python bench_serving/benchmark_serving.py \
    --backend vllm \
    --model <model_name> \
    --base-url http://localhost:8000 \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 1024 \
    --num-prompts 1000 \
    --max-concurrency 64 \
    --ignore-eos \
    --save-result
```

Supported backends: `vllm`, `sglang`, `tgi`, `tensorrt-llm`, `openai`, `openai-chat`, `lmdeploy`, `deepspeed-mii`, `scalellm`

## Key Metrics

Benchmarks measure:
- **TTFT**: Time to first token (ms) - prefill latency
- **TPOT**: Time per output token (ms, excluding first) - decode latency
- **Interactivity**: 1000/TPOT (tok/s/user) - user-perceived generation speed
- **Total Throughput**: Total tokens processed per second per GPU

## Visualizing Results

Use `plot_results.py` to generate benchmark plots:

```bash
# Activate virtual environment
source .venv/bin/activate

# Generate plot from results directory
python plot_results.py --results-dir results --output benchmark_plot.png
```

### Plot Features
- **Dual X-axis**: TTFT (top, orange) and Interactivity (bottom, blue)
- **Y-axis**: Total Throughput per GPU (tok/s)
- **Grouping**: Separate subplot per ISL/OSL combination
- **Labels**: Concurrency (c=X) shown on each data point

### Result File Naming Convention
Files must match pattern: `*_isl{ISL}_osl{OSL}_conc{CONC}_tp{TP}.json`

Example: `minimax_m2.1_vllm_isl512_osl8192_conc16_tp4.json`

### Required JSON Fields
- `median_tpot_ms`: Median time per output token
- `median_ttft_ms`: Median time to first token
- `total_token_throughput`: Total tokens/second
- `output_throughput`: Output tokens/second
