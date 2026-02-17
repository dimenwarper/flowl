"""Custom cost function: SoftDTW for time series.

Uses OTT-JAX's built-in ``SoftDTW`` cost to compare time series with
a transport metric that respects temporal warping.  Two groups of
synthetic time series (fast vs slow oscillations) are compared — SoftDTW
treats shifted/stretched versions as close, unlike Euclidean distance.
"""

import jax
import jax.numpy as jnp
from ott.geometry.costs import SoftDTW

from flowl import Cloud, Space, relax

# ── Synthetic time series data ──────────────────────────────────────
# Each "point" is a short time series (length 20), treated as a vector.
n_steps = 20
t = jnp.linspace(0, 2 * jnp.pi, n_steps)

key = jax.random.PRNGKey(42)


def make_series(key, freq, n_samples=15):
    """Generate n_samples sine waves with slight phase/amplitude jitter."""
    keys = jax.random.split(key, 3)
    phase = jax.random.uniform(keys[0], (n_samples, 1), minval=-0.3, maxval=0.3)
    amp = 1.0 + jax.random.uniform(keys[1], (n_samples, 1), minval=-0.15, maxval=0.15)
    noise = 0.05 * jax.random.normal(keys[2], (n_samples, n_steps))
    return amp * jnp.sin(freq * t[None, :] + phase) + noise


k1, k2 = jax.random.split(key)
slow_waves = make_series(k1, freq=1.0, n_samples=15)   # slow oscillation
fast_waves = make_series(k2, freq=2.5, n_samples=15)    # fast oscillation

print("Time Series — SoftDTW Custom Cost")
print("=" * 55)
print(f"  Slow waves: {slow_waves.shape[0]} series, {n_steps} steps each")
print(f"  Fast waves: {fast_waves.shape[0]} series, {n_steps} steps each")
print()

# ── Compare SoftDTW vs default (squared Euclidean) ──────────────────
results = {}

for label, geom in [("SoftDTW(gamma=1.0)", SoftDTW(gamma=1.0)), ("sq_euclidean", None)]:
    with Space(geometry=geom) as space:
        mu = Cloud("mu", n_particles=10, dim=n_steps)

        obs_slow = Cloud.from_samples("slow", slow_waves, fixed=True)
        obs_slow.covers(slow_waves)

        obs_fast = Cloud.from_samples("fast", fast_waves, fixed=True)
        obs_fast.covers(fast_waves)

        mu.drift(obs_slow, elasticity=1.0)
        mu.drift(obs_fast, elasticity=1.0)

        clouds = {"mu": mu, "slow": obs_slow, "fast": obs_fast}

    posterior = relax(space, clouds, lr=0.005, epsilon=0.1, max_steps=200)
    results[label] = posterior

print(f"{'Cost function':<25} {'Energy':>10} {'mu mean range':>15}")
print("-" * 55)

for label, post in results.items():
    mu_positions = post.positions("mu")
    mu_mean = jnp.mean(mu_positions, axis=0)
    val_range = float(jnp.max(mu_mean) - jnp.min(mu_mean))
    print(f"  {label:<23} {post.energy:>10.4f} {val_range:>15.4f}")

print()
print("SoftDTW is warp-invariant: it treats time-shifted versions of the")
print("same pattern as nearby, yielding a different energy landscape than")
print("the pointwise squared-Euclidean cost.")
