# ThrML Next.js + Three.js Demo

This example bootstraps a standalone Next.js application that visualizes key
metrics from the flagship ThrML thermodynamic algorithms using an interactive
Three.js scene. The page renders the stochastic resonance, active perception,
and Boltzmann policy planner outputs as rotating 3D clusters so stakeholders can
explore the KPIs in a tactile way.

## Prerequisites

- Node.js 18+
- npm 9+
- Python environment with ThrML installed (this repository)

## Generating fresh algorithm metrics

Before running the web demo, export the latest simulation results to the
application's static data folder:

```bash
python examples/build_nextjs_demo.py
```

The script will sample the SRSL, TAPS, and BPP algorithms with synthetic input
streams and emit `public/data/algorithm_metrics.json` for the frontend to load.

## Running the demo locally

```bash
cd examples/nextjs-demo
npm install
npm run dev
```

Then open http://localhost:3000 to explore the visualization. Click or tap any
node in the Three.js scene to lock onto its metrics—the details drawer beneath
the algorithm summaries updates live as you interact with the point cloud.

For production or CI builds use:

```bash
npm run build
npm start
```

The demo intentionally contains no backend dependencies—any static host that can
serve the compiled Next.js output will work.
