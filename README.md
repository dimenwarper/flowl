# flowl

Probabilistic programming via optimal transport.

Instead of parametric distributions and likelihoods, flowl uses particle-based **Clouds** connected by geometric **constraints** (drift, covers, warp). Inference is Wasserstein gradient flow — the solver relaxes a spring network to equilibrium. Built on JAX + OTT-JAX.

## Install

Requires Python 3.11+. Managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Quick start

```python
import jax.numpy as jnp
from flowl import Cloud, Space, relax

# 3-school hierarchical model
school_data = {
    "school_1": jnp.array([[-4.0], [-3.5], [-2.5], [-3.0], [-2.0]]),
    "school_2": jnp.array([[-0.5], [0.5], [0.0], [1.0], [-1.0]]),
    "school_3": jnp.array([[2.0], [3.5], [3.0], [4.0], [2.5]]),
}

with Space() as space:
    mu = Cloud("mu", n_particles=10, dim=1)       # global mean (free)
    clouds = {"mu": mu}

    for name, data in school_data.items():
        theta = Cloud(name, n_particles=len(data), dim=1)
        mu.drift(theta, elasticity=1.0)            # spring to global mean
        theta.covers(data)                         # anchor to observed data
        clouds[name] = theta

posterior = relax(space, clouds, lr=0.05, epsilon=0.5, max_steps=50)

print(posterior.mean("mu"))          # global mean estimate
print(posterior.mean("school_1"))    # shrunk toward global mean
```

## Core concepts

### Two-phase execution

1. **Tracing** — Python code inside `with Space():` builds a model graph (DAG of clouds and constraints).
2. **Solving** — `relax()` freezes the graph and runs JIT-compiled Wasserstein gradient flow via JAX.

### Clouds

A `Cloud` is a particle-based representation of a distribution. Each cloud has positions `(n, dim)` and weights `(n,)`.

- `Cloud(name, n_particles, dim)` — free cloud, initialized from downstream data
- `Cloud.from_samples(name, data, fixed=True)` — cloud fixed at observed positions

### Constraints

- **`drift(child, elasticity)`** — spring between parent and child clouds. Energy = `elasticity * sinkdiv(parent, child)`.
- **`covers(data)`** — anchors a cloud to observed data. Energy = `sinkdiv(cloud, data)`.
- **`WarpConstraint`** — transport-constrained mapping (Phase 2).

### Solver

- `relax(space, clouds, lr, epsilon, max_steps, tol)` — runs gradient descent on the total Sinkhorn divergence energy. Returns a `Posterior`.
- `Posterior` — access results via `posterior.mean(name)`, `posterior.std(name)`, `posterior.positions(name)`.

## Package structure

```
src/flowl/
├── core/           # Cloud, Space, constraints, model graph
├── tracing/        # Thread-local graph stack for model tracing
├── solver/         # Energy, JKO step (JIT-compiled), relax loop
├── backends/       # OTT-JAX wrappers (Sinkhorn divergence, OT coupling)
└── ops/            # Morph (displacement interpolation), Warp (Phase 2)
```

## Tests

```bash
uv run pytest
```
