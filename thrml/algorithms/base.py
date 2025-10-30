"""Base utilities and KPI tracking for thermal algorithms."""

import abc
from dataclasses import dataclass, field
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp


@dataclass
class KPITracker:
    """Tracks key performance indicators for thermal algorithms.
    
    **Attributes:**
    - `metrics`: Dictionary of metric name -> list of values
    - `timestamps`: List of timestamps for each measurement
    """
    
    metrics: dict[str, list[float]] = field(default_factory=dict)
    timestamps: list[float] = field(default_factory=list)
    
    def record(self, metric_name: str, value: float, timestamp: float | None = None):
        """Record a metric value."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(float(value))
        self.timestamps.append(timestamp if timestamp is not None else len(self.timestamps))
    
    def get_latest(self, metric_name: str) -> float | None:
        """Get the latest value for a metric."""
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return None
        return self.metrics[metric_name][-1]
    
    def get_mean(self, metric_name: str) -> float | None:
        """Get the mean value for a metric."""
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return None
        return sum(self.metrics[metric_name]) / len(self.metrics[metric_name])
    
    def clear(self):
        """Clear all recorded metrics."""
        self.metrics.clear()
        self.timestamps.clear()


class ThermalAlgorithm(eqx.Module):
    """Base class for thermal algorithms.
    
    All thermal algorithms should inherit from this and implement:
    - `forward()`: Main inference/processing step
    - `get_kpis()`: Return current KPI values
    """
    
    kpi_tracker: KPITracker = eqx.field(default_factory=KPITracker)
    
    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Run one step of the algorithm."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_kpis(self) -> dict[str, float]:
        """Return current KPI values."""
        raise NotImplementedError
    
    def reset_kpis(self):
        """Reset KPI tracking."""
        self.kpi_tracker.clear()


def mutual_information(x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Estimate mutual information I(X;Y) between two arrays.
    
    Uses histogram-based estimation for discrete data.
    """
    # Bin the data for MI estimation
    bins = 10
    hist_xy, _, _ = jnp.histogram2d(x.flatten(), y.flatten(), bins=bins)
    hist_x, _ = jnp.histogram(x.flatten(), bins=bins)
    hist_y, _ = jnp.histogram(y.flatten(), bins=bins)
    
    # Normalize to probabilities
    p_xy = hist_xy / jnp.sum(hist_xy) + 1e-10
    p_x = hist_x / jnp.sum(hist_x) + 1e-10
    p_y = hist_y / jnp.sum(hist_y) + 1e-10
    
    # Compute MI: I(X;Y) = sum(p(x,y) * log(p(x,y) / (p(x)*p(y))))
    p_x_expanded = jnp.expand_dims(p_x, 1)
    p_y_expanded = jnp.expand_dims(p_y, 0)
    mi = jnp.sum(p_xy * jnp.log(p_xy / (p_x_expanded * p_y_expanded) + 1e-10))
    return float(mi)


def entropy_from_samples(samples: jnp.ndarray) -> float:
    """Estimate entropy H(X) from samples using histogram."""
    bins = 20
    hist, _ = jnp.histogram(samples.flatten(), bins=bins)
    probs = hist / (jnp.sum(hist) + 1e-10)
    probs = probs[probs > 0]  # Remove zeros for log
    return float(-jnp.sum(probs * jnp.log(probs + 1e-10)))


def landauer_energy(entropy_delta: float, temperature_kelvin: float = 300.0) -> float:
    """Compute Landauer energy cost: E = kT * ln(2) * ΔH (in Joules).
    
    Args:
        entropy_delta: Change in entropy (bits)
        temperature_kelvin: Temperature in Kelvin (default 300K = room temp)
    
    Returns:
        Energy in Joules
    """
    k_boltzmann = 1.380649e-23  # J/K
    return k_boltzmann * temperature_kelvin * jnp.log(2.0) * entropy_delta


def beta_from_temperature(temperature: float) -> jnp.ndarray:
    """Convert temperature to inverse temperature beta.
    
    Args:
        temperature: Temperature value (larger = more thermal noise)
    
    Returns:
        Beta = 1/temperature
    """
    return jnp.array(1.0 / max(temperature, 1e-6))
