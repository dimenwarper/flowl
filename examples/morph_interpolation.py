"""Morph interpolation: trajectory between two point clouds.

Generates two crescent-shaped populations (via sklearn make_moons)
and computes the displacement interpolation trajectory between them,
simulating how one population could continuously deform into the other
— akin to trajectory inference in single-cell biology.
"""

import jax.numpy as jnp
from sklearn.datasets import make_moons

from flowl import Cloud, CloudState, Morph, Space, relax

# ── Data: two crescents ─────────────────────────────────────────────
X, labels = make_moons(n_samples=200, noise=0.08, random_state=42)
X = jnp.array(X, dtype=jnp.float32)

source_pts = X[labels == 0]  # upper crescent
target_pts = X[labels == 1]  # lower crescent

print("Morph Interpolation — Two Moons")
print("=" * 50)
print(f"  Source (upper moon): {source_pts.shape[0]} points")
print(f"  Target (lower moon): {target_pts.shape[0]} points")
print()

# ── Solve for the posterior positions ───────────────────────────────
with Space() as space:
    src = Cloud.from_samples("source", source_pts, fixed=True)
    src.covers(source_pts)

    tgt = Cloud.from_samples("target", target_pts, fixed=True)
    tgt.covers(target_pts)

    clouds = {"source": src, "target": tgt}

posterior = relax(space, clouds, lr=0.01, epsilon=0.05, max_steps=100)

# ── Compute morph trajectory ────────────────────────────────────────
source_state = posterior["source"]
target_state = posterior["target"]

morph = Morph(source_state, target_state, epsilon=0.05)

print("Displacement interpolation at t = 0.0, 0.25, 0.5, 0.75, 1.0:")
print("-" * 50)

for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    positions = morph.at(t)
    centroid = jnp.mean(positions, axis=0)
    spread = jnp.std(positions, axis=0)
    print(
        f"  t={t:.2f}  centroid=({centroid[0]:+.3f}, {centroid[1]:+.3f})"
        f"  spread=({spread[0]:.3f}, {spread[1]:.3f})"
    )

print()
print("The centroid smoothly moves from the source to target distribution,")
print("following the optimal transport map rather than a naive linear path.")
