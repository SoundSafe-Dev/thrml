"""Comprehensive benchmark and report generation for thermal algorithms."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import jax
import numpy as np

from thrml.algorithms.synthetic_data import BenchConfig, run_all


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    algorithm_name: str
    kpis: Dict[str, float]
    runtime_seconds: float
    success: bool
    error_message: str = ""


@dataclass
class BenchmarkReport:
    """Full benchmark report with statistics."""
    config: Dict
    results: List[BenchmarkResult]
    summary: Dict[str, Dict[str, float]]  # algorithm -> {mean_kpi: value}
    timestamp: str
    total_runtime_seconds: float


def run_comprehensive_benchmark(
    steps: int = 10,
    n_runs: int = 3,
    warmup: int = 50,
    samples: int = 200,
    steps_per_sample: int = 2,
    seed_base: int = 42,
) -> BenchmarkReport:
    """Run comprehensive benchmark across all algorithms.
    
    Args:
        steps: Number of steps per algorithm
        n_runs: Number of runs per algorithm (for statistics)
        warmup: Sampling warmup steps
        samples: Number of samples per step
        steps_per_sample: Gibbs steps per sample
        seed_base: Base seed for reproducibility
    
    Returns:
        Comprehensive benchmark report
    """
    start_time = time.time()
    
    config = {
        "steps": steps,
        "n_runs": n_runs,
        "warmup": warmup,
        "samples": samples,
        "steps_per_sample": steps_per_sample,
        "seed_base": seed_base,
    }
    
    all_results: List[BenchmarkResult] = []
    
    print(f"Running comprehensive benchmark: {n_runs} runs × 10 algorithms × {steps} steps")
    print("=" * 70)
    
    # Run each algorithm multiple times
    for run_idx in range(n_runs):
        seed = seed_base + run_idx * 1000
        cfg = BenchConfig(
            steps=steps,
            warmup=warmup,
            samples=samples,
            steps_per_sample=steps_per_sample,
            seed=seed,
        )
        
        print(f"\nRun {run_idx + 1}/{n_runs} (seed={seed})")
        print("-" * 70)
        
        results = run_all(cfg)
        
        for algo_name, kpis in results.items():
            runtime = 0.0  # Would need timing per algorithm
            result = BenchmarkResult(
                algorithm_name=algo_name,
                kpis=kpis,
                runtime_seconds=runtime,
                success=True,
            )
            all_results.append(result)
            print(f"  ✓ {algo_name}: {len(kpis)} KPIs recorded")
    
    # Compute statistics
    algo_stats: Dict[str, Dict[str, List[float]]] = {}
    for result in all_results:
        if result.algorithm_name not in algo_stats:
            algo_stats[result.algorithm_name] = {}
        for kpi_name, kpi_value in result.kpis.items():
            if kpi_name not in algo_stats[result.algorithm_name]:
                algo_stats[result.algorithm_name][kpi_name] = []
            algo_stats[result.algorithm_name][kpi_name].append(kpi_value)
    
    # Compute means
    summary: Dict[str, Dict[str, float]] = {}
    for algo_name, kpi_dict in algo_stats.items():
        summary[algo_name] = {}
        for kpi_name, values in kpi_dict.items():
            summary[algo_name][f"mean_{kpi_name}"] = float(np.mean(values))
            summary[algo_name][f"std_{kpi_name}"] = float(np.std(values))
            summary[algo_name][f"min_{kpi_name}"] = float(np.min(values))
            summary[algo_name][f"max_{kpi_name}"] = float(np.max(values))
    
    total_runtime = time.time() - start_time
    
    report = BenchmarkReport(
        config=config,
        results=all_results,
        summary=summary,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        total_runtime_seconds=total_runtime,
    )
    
    return report


def format_report(report: BenchmarkReport) -> str:
    """Format benchmark report as human-readable text."""
    lines = []
    lines.append("=" * 80)
    lines.append("THERMAL ALGORITHMS COMPREHENSIVE BENCHMARK REPORT")
    lines.append("=" * 80)
    lines.append(f"\nTimestamp: {report.timestamp}")
    lines.append(f"Total Runtime: {report.total_runtime_seconds:.2f} seconds")
    lines.append(f"\nConfiguration:")
    for key, value in report.config.items():
        lines.append(f"  {key}: {value}")
    
    lines.append("\n" + "=" * 80)
    lines.append("ALGORITHM PERFORMANCE SUMMARY")
    lines.append("=" * 80)
    
    # Group by algorithm
    algo_results = {}
    for result in report.results:
        if result.algorithm_name not in algo_results:
            algo_results[result.algorithm_name] = []
        algo_results[result.algorithm_name].append(result)
    
    for algo_name in sorted(report.summary.keys()):
        lines.append(f"\n{algo_name}")
        lines.append("-" * 80)
        
        # Show summary statistics
        for kpi_name, kpi_value in sorted(report.summary[algo_name].items()):
            if kpi_name.startswith("mean_"):
                base_name = kpi_name[5:]  # Remove "mean_" prefix
                mean_val = kpi_value
                std_val = report.summary[algo_name].get(f"std_{base_name}", 0.0)
                min_val = report.summary[algo_name].get(f"min_{base_name}", mean_val)
                max_val = report.summary[algo_name].get(f"max_{base_name}", mean_val)
                
                lines.append(f"  {base_name}:")
                lines.append(f"    Mean: {mean_val:.6f} ± {std_val:.6f}")
                lines.append(f"    Range: [{min_val:.6f}, {max_val:.6f}]")
        
        # Success rate
        algo_runs = algo_results[algo_name]
        success_count = sum(1 for r in algo_runs if r.success)
        success_rate = success_count / len(algo_runs) if algo_runs else 0.0
        lines.append(f"  Success Rate: {success_rate * 100:.1f}% ({success_count}/{len(algo_runs)} runs)")
    
    lines.append("\n" + "=" * 80)
    lines.append("KEY METRICS")
    lines.append("=" * 80)
    
    # Extract key metrics per algorithm
    key_metrics = {
        "SRSL": ["mean_signal_gain", "mean_mutual_information"],
        "TAPS": ["mean_threat_coverage", "mean_total_energy"],
        "BPP": ["mean_time_to_escalation", "mean_intervention_precision"],
        "EFSM": ["mean_anomaly_score", "mean_energy_delta"],
        "TBRO": ["mean_slo_compliance", "mean_gpu_hours_saved"],
        "LABI": ["mean_skip_rate", "mean_landauer_cost"],
        "TCF": ["mean_threat_score", "mean_robustness_delta"],
        "PPTS": ["mean_sync_error_ms", "mean_triangulation_error_m"],
        "TVS": ["mean_verified_rate", "mean_correlation"],
        "REF": ["mean_feature_stability", "mean_energy_per_feature"],
    }
    
    for algo_name, metrics in key_metrics.items():
        if algo_name in report.summary:
            lines.append(f"\n{algo_name}:")
            for metric in metrics:
                if metric in report.summary[algo_name]:
                    value = report.summary[algo_name][metric]
                    lines.append(f"  {metric}: {value:.6f}")
    
    lines.append("\n" + "=" * 80)
    
    return "\n".join(lines)


def save_report(report: BenchmarkReport, output_dir: Path = Path("benchmark_reports")):
    """Save benchmark report to files.
    
    Args:
        report: Benchmark report to save
        output_dir: Directory to save reports
    """
    output_dir.mkdir(exist_ok=True)
    
    timestamp_str = report.timestamp.replace(" ", "_").replace(":", "-")
    
    # Save JSON
    json_file = output_dir / f"benchmark_{timestamp_str}.json"
    with open(json_file, "w") as f:
        json.dump(
            {
                "config": report.config,
                "summary": report.summary,
                "timestamp": report.timestamp,
                "total_runtime_seconds": report.total_runtime_seconds,
            },
            f,
            indent=2,
            default=str,
        )
    
    # Save text report
    txt_file = output_dir / f"benchmark_{timestamp_str}.txt"
    with open(txt_file, "w") as f:
        f.write(format_report(report))
    
    # Save latest
    (output_dir / "benchmark_latest.json").symlink_to(json_file)
    (output_dir / "benchmark_latest.txt").symlink_to(txt_file)
    
    print(f"\nReports saved to:")
    print(f"  {json_file}")
    print(f"  {txt_file}")


def main():
    """Run comprehensive benchmark and generate report."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comprehensive thermal algorithms benchmark")
    parser.add_argument("--steps", type=int, default=10, help="Steps per algorithm")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs for statistics")
    parser.add_argument("--warmup", type=int, default=50, help="Sampling warmup steps")
    parser.add_argument("--samples", type=int, default=200, help="Samples per step")
    parser.add_argument("--output-dir", type=str, default="benchmark_reports", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    
    args = parser.parse_args()
    
    print("Initializing JAX...")
    jax.config.update("jax_enable_x64", True)
    
    print(f"\nRunning comprehensive benchmark:")
    print(f"  Steps: {args.steps}")
    print(f"  Runs: {args.runs}")
    print(f"  Warmup: {args.warmup}, Samples: {args.samples}")
    print(f"  Seed: {args.seed}\n")
    
    report = run_comprehensive_benchmark(
        steps=args.steps,
        n_runs=args.runs,
        warmup=args.warmup,
        samples=args.samples,
        seed_base=args.seed,
    )
    
    print("\n" + format_report(report))
    
    save_report(report, Path(args.output_dir))
    
    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()

