"""Time-constant conversions. Time constants are in seconds; dt is seconds per step."""
import math


def lam(tau, dt):
    """Per-step decay factor λ = exp(−dt/τ)."""
    return math.exp(-dt / tau)


def decay_over(tau, duration):
    """Decay factor over a gap of `duration` seconds, exp(−T/τ). Used for overnight breaks."""
    return math.exp(-duration / tau)


def to_steps(seconds, dt):
    """Number of whole steps closest to `seconds`."""
    return int(round(seconds / dt))


def log_spaced_taus(tau_min, tau_max, n):
    """n time constants log-spaced from tau_min to tau_max (seconds)."""
    if n == 1:
        return (float(tau_min),)
    r = math.log(tau_max / tau_min) / (n - 1)
    return tuple(tau_min * math.exp(r * k) for k in range(n))
