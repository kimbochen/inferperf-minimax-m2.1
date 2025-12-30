#!/usr/bin/env python3
"""
Visualization script for MiniMax M2.1 benchmark results.
Plots interactivity vs. throughput per GPU.
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt


@dataclass
class BenchmarkResult:
    isl: int
    osl: int
    max_conc: int
    tp: int
    # Interactivity (tok/s/user) - decode latency
    interactivity_median: float
    # TTFT (ms) - prefill latency
    ttft_median: float
    # Throughput per GPU (tok/s/GPU)
    total_throughput_per_gpu: float


def load_result(filepath: Path) -> BenchmarkResult:
    """Load a benchmark result JSON file."""
    pattern = r"isl(\d+)_osl(\d+)_conc(\d+)_tp(\d+)"
    match = re.search(pattern, filepath.name)
    if not match:
        raise ValueError(f"Could not parse filename: {filepath.name}")
    isl, osl, max_conc, tp = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))

    with open(filepath) as f:
        data = json.load(f)

    # Required fields - error if missing
    required_fields = ["median_tpot_ms", "median_ttft_ms",
                       "output_throughput", "total_token_throughput"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise KeyError(f"Missing required fields: {missing}")

    interactivity_median = 1000.0 / data["median_tpot_ms"]  # tok/s/user
    ttft_median = data["median_ttft_ms"]  # ms
    total_throughput_per_gpu = data["total_token_throughput"] / tp

    return BenchmarkResult(
        isl=isl,
        osl=osl,
        max_conc=max_conc,
        tp=tp,
        interactivity_median=interactivity_median,
        ttft_median=ttft_median,
        total_throughput_per_gpu=total_throughput_per_gpu,
    )


def load_and_group_results(results_dir: Path) -> Dict[Tuple[int, int], List[BenchmarkResult]]:
    """Load all benchmark results and group by ISL/OSL pair."""
    groups: Dict[Tuple[int, int], List[BenchmarkResult]] = {}
    for filepath in results_dir.glob("*.json"):
        try:
            r = load_result(filepath)
            key = (r.isl, r.osl)
            if key not in groups:
                groups[key] = []
            groups[key].append(r)
        except (ValueError, json.JSONDecodeError, KeyError) as e:
            print(f"Skipping {filepath.name}: {e}")
    # Sort each group by max_conc
    for key in groups:
        groups[key].sort(key=lambda x: x.max_conc)
    return groups


def plot_results(groups: Dict[Tuple[int, int], List[BenchmarkResult]], output_path: Path):
    """Create dual X-axis line chart: one subplot per ISL/OSL combo."""
    if not groups:
        print("No results to plot")
        return

    n_groups = len(groups)
    fig, axes = plt.subplots(n_groups, 1, figsize=(10, 6 * n_groups))
    if n_groups == 1:
        axes = [axes]

    # Colors for the two lines
    interactivity_color = '#1f77b4'  # blue
    ttft_color = '#ff7f0e'  # orange

    for idx, ((isl, osl), group) in enumerate(sorted(groups.items())):
        tp = group[0].tp

        # Extract data points
        interactivity = [r.interactivity_median for r in group]
        ttft = [r.ttft_median for r in group]
        throughput = [r.total_throughput_per_gpu for r in group]

        # Primary axis: TTFT (top X) vs Throughput (Y)
        ax1 = axes[idx]
        line1, = ax1.plot(
            ttft, throughput,
            's-', color=ttft_color, markersize=8, linewidth=2,
            label=f'TTFT TP{tp}'
        )
        # Label concurrency on TTFT line
        for i, r in enumerate(group):
            ax1.annotate(
                f"c={r.max_conc}",
                (ttft[i], throughput[i]),
                textcoords="offset points", xytext=(5, 5), fontsize=8,
                color=ttft_color
            )

        # Secondary axis: Interactivity (bottom X) vs Throughput (Y)
        ax2 = ax1.twiny()
        line2, = ax2.plot(
            interactivity, throughput,
            'o-', color=interactivity_color, markersize=8, linewidth=2,
            label=f'Interactivity TP{tp}'
        )
        # Label concurrency on Interactivity line
        for i, r in enumerate(group):
            ax2.annotate(
                f"c={r.max_conc}",
                (interactivity[i], throughput[i]),
                textcoords="offset points", xytext=(-25, -12), fontsize=8,
                color=interactivity_color
            )

        # Configure primary axis (top - TTFT)
        ax1.set_xlabel("TTFT (ms)", fontsize=11, color=ttft_color)
        ax1.tick_params(axis='x', labelcolor=ttft_color)
        ax1.xaxis.set_label_position('top')
        ax1.xaxis.tick_top()

        # Configure secondary axis (bottom - Interactivity)
        ax2.set_xlabel("Interactivity (tok/s/user)", fontsize=11, color=interactivity_color)
        ax2.tick_params(axis='x', labelcolor=interactivity_color)
        ax2.xaxis.set_label_position('bottom')
        ax2.xaxis.tick_bottom()

        # Y-axis (shared)
        ax1.set_ylabel("Total Throughput per GPU (tok/s)", fontsize=11)
        ax1.set_ylim(bottom=0)
        ax1.grid(True, alpha=0.3)

        # Title
        ax1.set_title(f"ISL:OSL = {isl}:{osl}", fontsize=13, pad=35)

        # Store lines for shared legend (only need one set)
        if idx == 0:
            legend_lines = [line1, line2]
            legend_labels = [f'TTFT TP{tp}', f'Interactivity TP{tp}']

    # Shared legend at figure level with concurrency note
    leg = fig.legend(legend_lines, legend_labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)
    # Add note explaining c=X labels
    fig.text(0.98, 0.96 - 0.04, "c = max concurrency", fontsize=9, ha='right', style='italic', color='gray')

    plt.suptitle("MiniMax M2.1 vLLM Benchmark", fontsize=16, ha='center', x=0.5)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(hspace=0.4)  # Add vertical space between graphs

    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Directory containing result JSON files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "benchmark_plot.png",
        help="Output plot filename"
    )
    args = parser.parse_args()

    print(f"Loading results from {args.results_dir}")
    groups = load_and_group_results(args.results_dir)
    total = sum(len(g) for g in groups.values())
    print(f"Loaded {total} results in {len(groups)} groups")

    if groups:
        plot_results(groups, args.output)
    else:
        print("No results found!")


if __name__ == "__main__":
    main()
