"""Interactive Gradio UI for THRML thermal algorithms."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
import jax
import jax.numpy as jnp

from thrml import SamplingSchedule
from thrml.algorithms import BoltzmannPolicyPlanner, StochasticResonanceSignalLifter


def _plot_signal_lifter(weak_features: jnp.ndarray, amplified: jnp.ndarray, optimal_beta: float) -> plt.Figure:
    """Plot weak vs amplified signals."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x_axis = np.arange(len(weak_features))
    ax.plot(x_axis, np.array(weak_features), label="Weak input", color="tab:gray")
    ax.plot(x_axis, np.array(amplified), label="Amplified (mean spin)", linestyle="--", color="tab:orange", linewidth=2)
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Activation")
    ax.set_title(f"Stochastic resonance sweep (optimal β = {optimal_beta:.2f})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _plot_policy_actions(action_probs: jnp.ndarray, action_names: list[str]) -> plt.Figure:
    """Bar chart of policy action probabilities."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(action_names, np.array(action_probs), color="tab:blue", alpha=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Policy action")
    ax.set_title("Boltzmann policy sampler")
    for bar, prob in zip(bars, np.array(action_probs)):
        ax.text(bar.get_x() + bar.get_width() / 2, prob + 0.02, f"{prob:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


def run_signal_lifter(
    seed: int,
    window_size: int,
    signal_frequency: float,
    signal_amplitude: float,
    noise_level: float,
    beta_min: float,
    beta_max: float,
    n_beta_steps: int,
    n_warmup: int,
    n_samples: int,
    steps_per_sample: int,
):
    """Generate a noisy waveform and amplify it with SRSL."""
    key = jax.random.key(int(seed))
    x = jnp.linspace(0, 2 * jnp.pi, window_size)
    clean_signal = signal_amplitude * jnp.sin(signal_frequency * x)
    noise = jax.random.normal(key, (window_size,)) * noise_level
    weak_features = clean_signal + noise

    srsl = StochasticResonanceSignalLifter(
        signal_window_size=window_size,
        beta_min=beta_min,
        beta_max=beta_max,
        n_beta_steps=n_beta_steps,
        key=key,
    )
    sampling_schedule = SamplingSchedule(n_warmup=n_warmup, n_samples=n_samples, steps_per_sample=steps_per_sample)
    amplified, kpis = srsl.forward(key, weak_features, sampling_schedule)

    fig = _plot_signal_lifter(weak_features, amplified, kpis["optimal_beta"])

    signal_power = float(jnp.mean(clean_signal**2))
    noise_power_in = float(jnp.mean(noise**2))
    noise_power_out = float(jnp.mean((amplified - clean_signal) ** 2))
    kpi_table = {
        "Optimal beta": round(float(kpis["optimal_beta"]), 4),
        "Mutual information": round(float(kpis["mutual_information"]), 4),
        "Signal gain": round(float(kpis["signal_gain"]), 4),
        "Energy per event (J)": f"{float(kpis['energy_per_event_joules']):.3e}",
        "Input SNR (dB)": round(10 * np.log10(signal_power / max(noise_power_in, 1e-9)), 2),
        "Amplified SNR (dB)": round(10 * np.log10(signal_power / max(noise_power_out, 1e-9)), 2),
        "Samples used": f"{n_samples} @ {steps_per_sample} steps (warmup {n_warmup})",
    }

    return fig, kpi_table


def run_policy_planner(
    seed: int,
    threat_level: int,
    tactic_0: float,
    tactic_1: float,
    tactic_2: float,
    tactic_3: float,
    tactic_4: float,
    risk_temperature: float,
):
    """Sample escalation policy probabilities from threat indicators."""
    tactic_scores = jnp.array([tactic_0, tactic_1, tactic_2, tactic_3, tactic_4])
    key = jax.random.key(int(seed))

    planner = BoltzmannPolicyPlanner(risk_temperature=risk_temperature, key=key)
    action_probs, kpis = planner.forward(key, threat_level=int(threat_level), tactic_scores=tactic_scores)

    fig = _plot_policy_actions(action_probs, planner.action_names)
    kpi_table = {
        "Selected action": kpis["selected_action"],
        "Action confidence": round(float(kpis["action_confidence"]), 4),
        "Time to escalation (proxy)": round(float(kpis["time_to_escalation"]), 4),
        "Precision": round(float(kpis["intervention_precision"]), 4),
        "Recall": round(float(kpis["intervention_recall"]), 4),
    }
    return fig, kpi_table


def build_demo() -> gr.Blocks:
    """Create the Gradio Blocks demo."""
    with gr.Blocks(title="THRML Thermal Algorithms", css=".gradio-container {max-width: 1100px; margin: auto;}") as demo:
        gr.Markdown(
            """
            # THRML Thermal Algorithms Live Demo

            Run two of the thermal algorithms end-to-end with live visuals:
            - **Stochastic-Resonance Signal Lifter (SRSL)** to amplify weak inputs.
            - **Boltzmann Policy Planner (BPP)** to sample escalation decisions from threat indicators.
            """
        )

        with gr.Tab("Signal Lifter"):
            with gr.Row():
                with gr.Column(scale=2):
                    seed = gr.Number(value=0, label="Random seed", precision=0)
                    window_size = gr.Slider(8, 64, value=32, step=1, label="Signal window size")
                    signal_frequency = gr.Slider(0.5, 4.0, value=1.0, step=0.1, label="Signal frequency (cycles)")
                    signal_amplitude = gr.Slider(0.1, 1.5, value=0.5, step=0.05, label="Signal amplitude")
                    noise_level = gr.Slider(0.01, 1.5, value=0.4, step=0.01, label="Noise level (std dev)")
                with gr.Column(scale=1):
                    beta_min = gr.Slider(0.05, 2.0, value=0.1, step=0.05, label="Beta sweep min")
                    beta_max = gr.Slider(0.5, 6.0, value=3.0, step=0.1, label="Beta sweep max")
                    n_beta_steps = gr.Slider(3, 40, value=12, step=1, label="Beta steps")
                    n_warmup = gr.Slider(0, 100, value=20, step=5, label="Warmup iterations")
                    n_samples = gr.Slider(10, 150, value=60, step=5, label="Samples")
                    steps_per_sample = gr.Slider(1, 10, value=2, step=1, label="Steps per sample")
                    run_button = gr.Button("Run SRSL", variant="primary")

            srsl_plot = gr.Plot(label="Signal lift visualization")
            srsl_kpis = gr.JSON(label="SRSL KPIs")
            run_button.click(
                run_signal_lifter,
                inputs=[
                    seed,
                    window_size,
                    signal_frequency,
                    signal_amplitude,
                    noise_level,
                    beta_min,
                    beta_max,
                    n_beta_steps,
                    n_warmup,
                    n_samples,
                    steps_per_sample,
                ],
                outputs=[srsl_plot, srsl_kpis],
            )

        with gr.Tab("Policy Planner"):
            with gr.Row():
                with gr.Column(scale=2):
                    threat_level = gr.Slider(0, 2, value=1, step=1, label="Threat level (0=low, 1=medium, 2=high)")
                    tactic_0 = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="Tactic: loitering")
                    tactic_1 = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Tactic: crowd density")
                    tactic_2 = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Tactic: perimeter breach")
                    tactic_3 = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="Tactic: suspicious object")
                    tactic_4 = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Tactic: audio threat")
                with gr.Column(scale=1):
                    risk_temperature = gr.Slider(0.3, 2.5, value=1.0, step=0.05, label="Risk temperature (higher = conservative)")
                    planner_seed = gr.Number(value=7, label="Random seed", precision=0)
                    plan_button = gr.Button("Sample policy", variant="primary")

            bpp_plot = gr.Plot(label="Policy action probabilities")
            bpp_kpis = gr.JSON(label="Policy KPIs")
            plan_button.click(
                run_policy_planner,
                inputs=[
                    planner_seed,
                    threat_level,
                    tactic_0,
                    tactic_1,
                    tactic_2,
                    tactic_3,
                    tactic_4,
                    risk_temperature,
                ],
                outputs=[bpp_plot, bpp_kpis],
            )

        gr.Markdown(
            """
            Tip: sweep the **risk temperature** to see how the policy sampler flips from decisive (low temperature) to cautious
            (high temperature), and raise the **noise level** on the signal lifter to watch stochastic resonance pull the weak
            pattern back above the detection threshold.
            """
        )

        demo.load(
            run_signal_lifter,
            inputs=[
                0,
                32,
                1.0,
                0.5,
                0.4,
                0.1,
                3.0,
                12,
                20,
                60,
                2,
            ],
            outputs=[srsl_plot, srsl_kpis],
        )

        demo.load(
            run_policy_planner,
            inputs=[
                7,
                1,
                0.6,
                0.3,
                0.5,
                0.4,
                0.2,
                1.0,
            ],
            outputs=[bpp_plot, bpp_kpis],
        )

    return demo


def main():
    """Launch the Gradio app."""
    demo = build_demo()
    demo.queue(concurrency_count=4)
    demo.launch()


if __name__ == "__main__":
    main()
