"""Morph interpolation: Euclidean vs Hyperbolic trajectory comparison.

Generates two-moons data scaled inside the Poincaré disk, runs morph
interpolation under both Euclidean and Hyperbolic geometries, and produces
a side-by-side animation saved as ``examples/morph_comparison.gif``.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from flowl import Cloud, Morph, Space, relax
from flowl.core.geometry import resolve_geometry

# ── Data: two clusters at different radii → boundary ring ────────
# Source: one cluster near center + one cluster near boundary
# Target: ring of points near the boundary
# Hyperbolic distance heavily penalises boundary→boundary hops,
# so the coupling (which source goes where) changes dramatically.
key = jax.random.PRNGKey(42)
k1, k2, k3 = jax.random.split(key, 3)

n_half = 40
n_pts = 2 * n_half

# Source cluster A: near center
src_a = 0.12 * jax.random.normal(k1, (n_half, 2))
# Source cluster B: near top-right boundary
src_b = jnp.array([0.75, 0.45]) + 0.08 * jax.random.normal(k2, (n_half, 2))
source_pts = jnp.concatenate([src_a, src_b], axis=0)

# Target: ring near boundary (radius ~0.9)
angles = jnp.linspace(0, 2 * jnp.pi, n_pts, endpoint=False)
radii = 0.90 + 0.03 * jax.random.normal(k3, (n_pts,))
target_pts = jnp.stack([radii * jnp.cos(angles), radii * jnp.sin(angles)], axis=1)

# Clamp into the Poincaré disk (max radius 0.98)
def clamp_to_disk(pts, max_r=0.98):
    norms = jnp.linalg.norm(pts, axis=1, keepdims=True)
    scale = jnp.minimum(1.0, max_r / jnp.maximum(norms, 1e-8))
    return pts * scale

source_pts = clamp_to_disk(jnp.array(source_pts, dtype=jnp.float32))
target_pts = clamp_to_disk(jnp.array(target_pts, dtype=jnp.float32))

print("Morph Interpolation — Euclidean vs Hyperbolic")
print("=" * 55)
print(f"  Source: {source_pts.shape[0]} points (center + boundary clusters)")
print(f"  Target: {target_pts.shape[0]} points (boundary ring, r~0.90)")
all_pts = jnp.concatenate([source_pts, target_pts])
print(f"  Max ||x||: {float(jnp.max(jnp.linalg.norm(all_pts, axis=1))):.3f}")
print()


# ── Solve for both geometries ──────────────────────────────────────
def solve_morph(geometry_name, geometry_arg):
    """Run relax + morph for a given geometry."""
    geo = resolve_geometry(geometry_arg)
    with Space(geometry=geometry_arg) as space:
        src = Cloud.from_samples("source", source_pts, fixed=True)
        src.covers(source_pts)
        tgt = Cloud.from_samples("target", target_pts, fixed=True)
        tgt.covers(target_pts)
        clouds = {"source": src, "target": tgt}

    posterior = relax(space, clouds, lr=0.01, epsilon=0.001, max_steps=500)
    morph = Morph(posterior["source"], posterior["target"], geometry=geo, epsilon=0.001)

    ts = jnp.linspace(0.0, 1.0, 30)
    frames = [morph.at(float(t)) for t in ts]

    print(f"  [{geometry_name}] morph computed — {len(frames)} frames")
    return frames, ts


print("Computing morph trajectories...")
frames_euc, ts = solve_morph("Euclidean", None)
frames_hyp, _ = solve_morph("Hyperbolic", "hyperbolic")

# ── Summary stats ──────────────────────────────────────────────────
print()
print("Displacement interpolation centroids:")
print("-" * 55)
print(f"  {'t':>4s}  {'Euclidean':>20s}  {'Hyperbolic':>20s}")
for i, t in enumerate(ts):
    if float(t) not in (0.0, 0.25, 0.5, 0.75, 1.0):
        # Only print a few keyframes
        if not any(abs(float(t) - k) < 0.02 for k in [0.0, 0.25, 0.5, 0.75, 1.0]):
            continue
    ce = jnp.mean(frames_euc[i], axis=0)
    ch = jnp.mean(frames_hyp[i], axis=0)
    print(
        f"  {float(t):.2f}  ({ce[0]:+.3f}, {ce[1]:+.3f})"
        f"  ({ch[0]:+.3f}, {ch[1]:+.3f})"
    )

print()

# ── Animation ──────────────────────────────────────────────────────
fig, (ax_euc, ax_hyp) = plt.subplots(1, 2, figsize=(10, 5))

for ax, title in [(ax_euc, "Euclidean"), (ax_hyp, "Hyperbolic")]:
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14)
    circle = plt.Circle((0, 0), 1.0, fill=False, color="gray", ls="--", lw=0.8)
    ax.add_patch(circle)
    # Plot original source/target moons as background reference
    ax.scatter(source_pts[:, 0], source_pts[:, 1], s=8, c="steelblue", alpha=0.15, label="source")
    ax.scatter(target_pts[:, 0], target_pts[:, 1], s=8, c="coral", alpha=0.15, label="target")

scat_euc = ax_euc.scatter([], [], s=10, c="steelblue", alpha=0.7)
scat_hyp = ax_hyp.scatter([], [], s=10, c="coral", alpha=0.7)
time_text = fig.suptitle("t = 0.00", fontsize=13)

# Build frame sequence: hold first and last frames longer
HOLD = 8
frame_seq = [0] * HOLD + list(range(len(ts))) + [len(ts) - 1] * HOLD


def update(seq_idx):
    frame_idx = frame_seq[seq_idx]
    pts_e = frames_euc[frame_idx]
    pts_h = frames_hyp[frame_idx]
    scat_euc.set_offsets(pts_e)
    scat_hyp.set_offsets(pts_h)
    time_text.set_text(f"t = {float(ts[frame_idx]):.2f}")
    return scat_euc, scat_hyp, time_text


ani = animation.FuncAnimation(
    fig, update, frames=len(frame_seq), interval=100, blit=True
)

out_path = os.path.join(os.path.dirname(__file__), "morph_comparison.gif")
ani.save(out_path, writer="pillow", fps=10)
plt.close(fig)
print(f"Animation saved to {out_path}")
