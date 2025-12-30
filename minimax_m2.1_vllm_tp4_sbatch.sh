#!/usr/bin/env bash
#SBATCH -p main
#SBATCH --gres=gpu:4
#SBATCH --container-image=vllm/vllm-openai:nightly-8711b216766bb5d3cbe15161061c3a7d9fffe59c
#SBATCH --container-mounts=/home/kimbo/inferperf_minimax-m2.1/model_weights:/model_weights,/home/kimbo/inferperf_minimax-m2.1/bench_serving:/bench_serving,/home/kimbo/inferperf_minimax-m2.1/results:/results
#SBATCH --no-container-entrypoint

# Benchmark script for MiniMax M2.1 with vLLM on Slurm using pyxis
# Runs all combinations of input/output lengths and concurrency levels
# Usage: sbatch minimax_m2.1_vllm_tp4_sbatch.sh

set -euo pipefail

MODEL="MiniMaxAI/MiniMax-M2.1"
PORT=8000

# Start vLLM server in background
SERVER_LOG=/results/vllm-server.log
vllm serve $MODEL \
    --download-dir /model_weights \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think \
    --enable-auto-tool-choice \
    --host 0.0.0.0 \
    --port $PORT \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "Server PID: $SERVER_PID, log: $SERVER_LOG"

# Wait for server to be ready
echo "Waiting for vLLM server to be ready..."
tail -f -n +1 "$SERVER_LOG" &
TAIL_PID=$!

until curl --output /dev/null --silent --fail http://0.0.0.0:$PORT/health; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server died before becoming healthy. Exiting."
        kill $TAIL_PID 2>/dev/null || true
        cat "$SERVER_LOG"
        exit 1
    fi
    sleep 5
done
kill $TAIL_PID 2>/dev/null || true
echo "Server is ready!"

# === Benchmark configurations ===
INPUT_OUTPUT_COMBOS=(
    '512 8192'
    '8192 512'
)
MAX_CONCS=(4 8 16 24 32 40)

pip install -q pandas datasets

# === Run all combinations ===
for combo in "${INPUT_OUTPUT_COMBOS[@]}"; do
    read -r INPUT_LEN OUTPUT_LEN <<< "$combo"
    for MAX_CONC in "${MAX_CONCS[@]}"; do
        NUM_WARMUPS=$((MAX_CONC * 2))
        NUM_PROMPTS=$((MAX_CONC * 10))
        RESULT_FILENAME="minimax_m2.1_vllm_isl${INPUT_LEN}_osl${OUTPUT_LEN}_conc${MAX_CONC}_tp4.json"

        echo "=== MiniMax M2.1 vLLM Benchmark ==="
        echo "Input Length: $INPUT_LEN"
        echo "Output Length: $OUTPUT_LEN"
        echo "Max Concurrency: $MAX_CONC"
        echo "Num Warmups: $NUM_WARMUPS"
        echo "Num Prompts: $NUM_PROMPTS"
        echo "Result File: $RESULT_FILENAME"
        echo "===================================="

        python3 /bench_serving/benchmark_serving.py \
            --model $MODEL \
            --backend openai \
            --base-url http://0.0.0.0:$PORT \
            --dataset-name random \
            --random-input-len $INPUT_LEN \
            --random-output-len $OUTPUT_LEN \
            --random-range-ratio 0.2 \
            --num-prompts $NUM_PROMPTS \
            --max-concurrency $MAX_CONC \
            --num-warmups $NUM_WARMUPS \
            --request-rate inf \
            --ignore-eos \
            --save-result \
            --percentile-metrics 'ttft,tpot,itl,e2el' \
            --result-dir /results \
            --result-filename $RESULT_FILENAME

        echo "Benchmark complete for ISL ${INPUT_LEN} OSL ${OUTPUT_LEN} CONC ${MAX_CONC}"
    done
done

# Cleanup
kill $SERVER_PID 2>/dev/null || true
