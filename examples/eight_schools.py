"""Eight Schools: hierarchical Bayesian shrinkage via optimal transport.

The classic dataset from Rubin (1981) — eight schools each report an
estimated SAT coaching effect and its standard error.  A hierarchical
model lets the estimates borrow strength from one another, shrinking
noisy extremes toward the group mean.

This example uses the default squared-Euclidean geometry.
"""

import jax.numpy as jnp

from flowl import Cloud, Space, relax

# ── Data (Rubin 1981) ──────────────────────────────────────────────
#   school:  A    B    C    D    E    F    G    H
effects = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
std_errs = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
school_names = list("ABCDEFGH")

n_particles = 30

with Space() as space:
    # Population-level latent (the "group mean")
    mu = Cloud("mu", n_particles=n_particles, dim=1)
    clouds = {"mu": mu}

    for i, (y, se) in enumerate(zip(effects, std_errs)):
        name = school_names[i]
        # Spread observed data around the point estimate ± 2 SE
        samples = jnp.linspace(y - 2 * se, y + 2 * se, n_particles)

        # School-level latent (free) that gets pulled between mu and data
        theta = Cloud(name, n_particles=n_particles, dim=1)
        mu.drift(theta, elasticity=1.0)
        theta.covers(samples[:, None])
        clouds[name] = theta

posterior = relax(space, clouds, lr=0.05, epsilon=0.5, max_steps=200, tol=1e-5)

# ── Results ─────────────────────────────────────────────────────────
print("Eight Schools — Hierarchical Shrinkage")
print("=" * 55)
print(f"{'School':<8} {'Raw effect':>11} {'Posterior mean':>15} {'Shrinkage':>10}")
print("-" * 55)

pop_mean = float(posterior.mean("mu")[0])

for name, y in zip(school_names, effects):
    pm = float(posterior.mean(name)[0])
    shrinkage = 1.0 - abs(pm - pop_mean) / max(abs(y - pop_mean), 1e-6)
    print(f"  {name:<6} {y:>11.1f} {pm:>15.2f} {shrinkage:>9.0%}")

print("-" * 55)
print(f"  {'mu':<6} {'':>11} {pop_mean:>15.2f}")
print()
print(f"Final energy: {posterior.energy:.4f}")
print()
print("Shrinkage = how far the posterior moved toward the population mean.")
print("Schools with noisier estimates (large SE) shrink more.")
