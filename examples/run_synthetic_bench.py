#!/usr/bin/env python3
"""Run synthetic benchmarks for all 10 thermal algorithms.

Usage:
  python examples/run_synthetic_bench.py --steps 5 --seed 0

Prints a KPI summary per algorithm.
"""

import argparse
import json

from thrml.algorithms.synthetic_data import BenchConfig, run_all


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--steps", type=int, default=5)
	parser.add_argument("--warmup", type=int, default=50)
	parser.add_argument("--samples", type=int, default=100)
	parser.add_argument("--steps_per_sample", type=int, default=2)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--json", action="store_true", help="Print JSON only")
	args = parser.parse_args()

	cfg = BenchConfig(
		steps=args.steps,
		warmup=args.warmup,
		samples=args.samples,
		steps_per_sample=args.steps_per_sample,
		seed=args.seed,
	)

	results = run_all(cfg)

	if args.json:
		print(json.dumps(results, indent=2, default=float))
		return

	print("\nSynthetic KPI Summary")
	print("=" * 24)
	for name, kpis in results.items():
		print(f"\n{name}")
		for k, v in sorted(kpis.items()):
			print(f"  {k}: {v}")


if __name__ == "__main__":
	main()
