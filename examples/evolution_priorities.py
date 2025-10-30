#!/usr/bin/env python3
"""Interactive tool to prioritize evolution and expansion candidates.

Helps teams decide which algorithms to evolve first and which new categories to explore.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EVOLUTION_CANDIDATES = {
    "A-SRSL (Adaptive SRSL)": {
        "description": "Multi-band resonance, adaptive windows, temporal dynamics",
        "complexity": "Medium",
        "impact": "High",
        "dependencies": [],
        "use_cases": ["Multi-speaker detection", "Long streams", "Non-stationary signals"],
    },
    "LA-TAPS (Learning-Aware TAPS)": {
        "description": "Threat model learning, dynamic bitrate, multi-objective",
        "complexity": "High",
        "impact": "Very High",
        "dependencies": [],
        "use_cases": ["Adaptive networks", "Budget constraints", "Time-varying threats"],
    },
    "E-BPP (Ensemble BPP)": {
        "description": "Multi-policy ensemble, RL integration, context-aware",
        "complexity": "High",
        "impact": "High",
        "dependencies": [],
        "use_cases": ["Complex ROE", "Learning from history", "Multi-domain"],
    },
    "MB-EFSM (Multi-Baseline EFSM)": {
        "description": "Ensemble baselines, drift adaptation, temporal baselines",
        "complexity": "Medium",
        "impact": "High",
        "dependencies": ["Fix EFSM broadcasting"],
        "use_cases": ["Multi-mode ops", "Environment changes", "Pattern discovery"],
    },
    "P-TBRO (Predictive TBRO)": {
        "description": "Adaptive priors, multi-horizon, predictive modeling",
        "complexity": "Medium",
        "impact": "Medium",
        "dependencies": [],
        "use_cases": ["Proactive management", "Event scaling", "Hierarchical alloc"],
    },
    "G-LABI (Graduated LABI)": {
        "description": "Graduated updates, adaptive thresholds, multi-scale",
        "complexity": "Low",
        "impact": "Medium",
        "dependencies": [],
        "use_cases": ["Fine-grained control", "Adaptive energy", "Multi-resolution"],
    },
}

NEW_CATEGORIES = {
    "Temporal & Sequential": {
        "algorithms": ["TSM", "TPC", "TTAD"],
        "description": "Time-series and sequence processing",
        "priority": "High",
    },
    "Distributed & Federated": {
        "algorithms": ["FTL", "TCP", "DRA"],
        "description": "Multi-node coordination and learning",
        "priority": "Medium",
    },
    "Multi-Modal Fusion": {
        "algorithms": ["CMTA", "TLAF", "MMEL"],
        "description": "Advanced fusion capabilities",
        "priority": "Very High",
    },
    "Advanced Optimization": {
        "algorithms": ["TCO", "MOTP", "TML"],
        "description": "Constrained and multi-objective optimization",
        "priority": "Medium",
    },
    "Security & Adversarial": {
        "algorithms": ["TAD", "TZKP", "TDP"],
        "description": "Security and privacy algorithms",
        "priority": "High",
    },
}


def generate_prioritization_template(outdir: Path):
    """Generate prioritization template for team voting."""
    template = {
        "evolution_candidates": EVOLUTION_CANDIDATES,
        "new_categories": NEW_CATEGORIES,
        "scoring_criteria": {
            "impact": "How much does this improve operational wins?",
            "feasibility": "How easy is this to implement?",
            "urgency": "How urgently is this needed?",
            "dependencies": "What blockers exist?",
        },
        "scoring_scale": {
            "impact": ["Low", "Medium", "High", "Very High"],
            "feasibility": ["Easy", "Medium", "Hard", "Very Hard"],
            "urgency": ["Nice to have", "Should have", "Must have", "Critical"],
        },
    }
    
    (outdir / "prioritization_template.json").write_text(
        json.dumps(template, indent=2)
    )
    print(f"✓ Prioritization template saved to {outdir / 'prioritization_template.json'}")


def generate_roadmap_summary(outdir: Path):
    """Generate markdown summary of roadmap."""
    summary = f"""# Evolution & Expansion Roadmap Summary

## Evolution Candidates (Phase 1)

### Top Priority Candidates

1. **LA-TAPS (Learning-Aware TAPS)** ⭐⭐⭐
   - Impact: Very High
   - Complexity: High
   - Use cases: Adaptive networks, budget constraints
   - **Why first**: High ROI, addresses key limitation

2. **A-SRSL (Adaptive SRSL)** ⭐⭐
   - Impact: High
   - Complexity: Medium
   - Use cases: Multi-speaker, long streams
   - **Why second**: Manageable scope, clear improvements

3. **MB-EFSM (Multi-Baseline EFSM)** ⭐⭐
   - Impact: High
   - Complexity: Medium
   - Dependencies: Fix current EFSM
   - Use cases: Multi-mode operations
   - **Why third**: Fixes current issues while adding value

### Secondary Candidates

4. **E-BPP (Ensemble BPP)**: Complex but high value
5. **P-TBRO (Predictive TBRO)**: Medium impact, manageable
6. **G-LABI (Graduated LABI)**: Lower complexity, incremental

## New Categories (Phase 2)

### Must-Have Categories

1. **Multi-Modal Fusion** (Very High Priority)
   - CMTA, TLAF, MMEL
   - Critical for SoundSafe advanced use cases

2. **Temporal & Sequential** (High Priority)
   - TSM, TPC, TTAD
   - Essential for audio/video streams

3. **Security & Adversarial** (High Priority)
   - TAD, TZKP, TDP
   - Needed for production security

### Nice-to-Have Categories

4. **Distributed & Federated** (Medium Priority)
5. **Advanced Optimization** (Medium Priority)

## Recommended Timeline

### Q1 2026
- Fix EFSM broadcasting issues
- Implement A-SRSL
- Begin LA-TAPS design

### Q2 2026
- Complete LA-TAPS
- Implement MB-EFSM
- Add CMTA (multi-modal)

### Q3 2026
- Add TSM (temporal)
- Implement E-BPP
- Begin integration framework

### Q4 2026
- Complete integration framework
- Add 2-3 more Phase 2 algorithms
- Production deployment prep

## Decision Framework

When prioritizing, consider:
- **ROI**: Impact × Feasibility
- **Dependencies**: Blockers and prerequisites
- **Strategic fit**: Alignment with SoundSafe roadmap
- **Extropic readiness**: Hardware optimization potential

## Next Actions

1. Team voting on prioritization template
2. Detailed design specs for top 3
3. Prototype development
4. Validation and benchmarking
5. Production integration

---

See `docs/EVOLUTION_ROADMAP.md` for complete details.
"""
    
    (outdir / "roadmap_summary.md").write_text(summary)
    print(f"✓ Roadmap summary saved to {outdir / 'roadmap_summary.md'}")


def main():
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--output", type=str, default="results/evolution_roadmap")
    args = parser.parse_args()
    
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    
    generate_prioritization_template(outdir)
    generate_roadmap_summary(outdir)
    
    print(f"\n✓ Evolution roadmap materials generated in: {outdir}")


if __name__ == "__main__":
    main()

