import pytest
import jax
import jax.numpy as jnp

from thrml import SamplingSchedule, Block
from thrml.algorithms import (
    StochasticResonanceSignalLifter,
    ThermodynamicActivePerceptionScheduling,
    BoltzmannPolicyPlanner,
    EnergyFingerprintedSceneMemory,
    ThermalBanditResourceOrchestrator,
    LandauerAwareBayesianInference,
    ThermodynamicCausalFusion,
    ProbabilisticPhaseTimeSync,
    ThermoVerifiableSensing,
    ReservoirEBMFrontEnd,
)

FAST = SamplingSchedule(n_warmup=5, n_samples=10, steps_per_sample=1)


def test_srsl_smoke():
    key = jax.random.key(0)
    srsl = StochasticResonanceSignalLifter(signal_window_size=16, key=key)
    weak = jax.random.normal(key, (16,)) * 0.2
    _, kpis = srsl.forward(key, weak, FAST)
    assert "mutual_information" in kpis


def test_taps_smoke():
    key = jax.random.key(1)
    taps = ThermodynamicActivePerceptionScheduling(n_sensors=6, n_bitrate_levels=3, key=key)
    threat = jax.random.uniform(key, (6,))
    _, kpis = taps.forward(key, threat, FAST)
    assert kpis["total_energy_joules_per_sec"] >= 0.0


def test_bpp_smoke():
    key = jax.random.key(2)
    bpp = BoltzmannPolicyPlanner(risk_temperature=1.0, key=key)
    tactics = jnp.array([0.1, 0.5, 0.2, 0.7, 0.3])
    _, kpis = bpp.forward(key, 1, tactics, FAST)
    assert kpis["selected_action"] in ("notify", "lock", "dispatch", "no_action")


@pytest.mark.xfail(reason="EFSM broadcasting in energy eval path under investigation")
def test_efsm_smoke():
    key = jax.random.key(3)
    efsm = EnergyFingerprintedSceneMemory(n_features=32, key=key)
    clean = jax.random.normal(key, (20, 32)) * 0.2
    efsm.fit_baseline(key, clean, FAST)
    scene = jax.random.normal(key, (32,)) * 0.2
    _, kpis = efsm.forward(key, scene, FAST)
    assert "anomaly_score" in kpis


def test_tbro_smoke():
    key = jax.random.key(4)
    tbro = ThermalBanditResourceOrchestrator(n_sites=6, key=key)
    risk = jax.random.uniform(key, (6,))
    _, kpis = tbro.forward(key, risk, FAST)
    assert "slo_compliance" in kpis


def test_labi_smoke():
    key = jax.random.key(5)
    labi = LandauerAwareBayesianInference(n_variables=12, energy_threshold=1e-19, key=key)
    lh = jax.random.uniform(key, (12,), minval=-0.1, maxval=0.1)
    (_, _), kpis = labi.forward(key, lh, FAST)
    assert 0.0 <= kpis["skip_rate"] <= 1.0


def test_tcf_smoke():
    key = jax.random.key(6)
    tcf = ThermodynamicCausalFusion(n_modalities=4, key=key)
    mods = jax.random.uniform(key, (4,))
    _, k1 = tcf.discover_causal_structure(key, mods, schedule=FAST)
    _, k2 = tcf.forward(key, mods, schedule=FAST)
    assert "n_causal_edges" in k1 and "threat_score" in k2


def test_ppts_smoke():
    key = jax.random.key(7)
    ppts = ProbabilisticPhaseTimeSync(n_sensors=4, key=key)
    obs = jax.random.uniform(key, (4,), minval=0.0, maxval=2 * jnp.pi)
    _, kpis = ppts.forward(key, obs, FAST)
    assert kpis["sync_error_ms"] >= 0.0


def test_tvs_smoke():
    key = jax.random.key(8)
    tvs = ThermoVerifiableSensing(nonce_size=16, watermark_strength=0.05, key=key)
    stream = jax.random.normal(key, (40, 32))
    wm, k1 = tvs.forward(key, stream, mode="watermark", schedule=FAST)
    ok, k2 = tvs.verify_stream(jnp.array(wm))
    assert "bitrate_overhead_percent" in k1 and "correlation" in k2


def test_ref_smoke():
    key = jax.random.key(9)
    ref = ReservoirEBMFrontEnd(reservoir_size=48, feature_size=16, key=key)
    raw = jax.random.normal(key, (20, 12))
    _, kpis = ref.forward(key, raw, FAST)
    assert "feature_stability" in kpis
