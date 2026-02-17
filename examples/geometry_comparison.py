"""Geometry comparison: same data, different metrics.

Uses the UCI Wine dataset (178 samples, 13 chemical features, 3 cultivars)
to show how the choice of geometry affects the posterior.  Features have
very different scales (e.g. proline ~1000, color intensity ~5), so
euclidean vs cosine vs p-norm give meaningfully different results.
"""

import jax.numpy as jnp
from ott.geometry.costs import PNormP
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

from flowl import Cloud, Space, relax

# ── Data ────────────────────────────────────────────────────────────
wine = load_wine()
X = jnp.array(StandardScaler().fit_transform(wine.data), dtype=jnp.float32)
labels = wine.target  # 0, 1, 2

# Pick two cultivars
class_a = X[labels == 0]  # ~59 samples
class_b = X[labels == 1]  # ~71 samples

geometries = {
    "sq_euclidean (default)": None,
    "euclidean": "euclidean",
    "cosine": "cosine",
    "PNormP(p=1.5)": PNormP(p=1.5),
}

# ── Run each geometry ───────────────────────────────────────────────
print("Wine Dataset — Geometry Comparison")
print("=" * 65)
print(f"  Class 0: {class_a.shape[0]} samples, Class 1: {class_b.shape[0]} samples")
print(f"  Features: {X.shape[1]} (standardized)")
print()
print(f"{'Geometry':<25} {'Energy':>10} {'Posterior mean norm (mu)':>25}")
print("-" * 65)

for name, geom in geometries.items():
    with Space(geometry=geom) as space:
        mu = Cloud("mu", n_particles=40, dim=X.shape[1])

        obs_a = Cloud.from_samples("obs_a", class_a, fixed=True)
        obs_a.covers(class_a)

        obs_b = Cloud.from_samples("obs_b", class_b, fixed=True)
        obs_b.covers(class_b)

        mu.drift(obs_a, elasticity=1.0)
        mu.drift(obs_b, elasticity=1.0)

        clouds = {"mu": mu, "obs_a": obs_a, "obs_b": obs_b}

    posterior = relax(space, clouds, lr=0.005, epsilon=0.1, max_steps=200)

    mu_mean = posterior.mean("mu")
    mu_norm = float(jnp.linalg.norm(mu_mean))

    print(f"  {name:<23} {posterior.energy:>10.4f} {mu_norm:>25.4f}")

print()
print("Different geometries yield different energies and posterior locations,")
print("because each metric weighs feature dimensions differently.")
