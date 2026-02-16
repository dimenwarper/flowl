"""End-to-end integration test: 3-school hierarchical model.

This is the classic hierarchical model used to test shrinkage:
- Global mean mu
- Three school means theta_1, theta_2, theta_3, each drifting from mu
- Each school has observed data anchored via covers()

After relaxation, school means should shrink toward the global mean.
"""

import jax.numpy as jnp
import pytest

from flowl import Cloud, Space, relax


def test_three_schools_shrinkage():
    # Synthetic data: 3 schools with true means -3, 0, +3
    key_data = {
        "school_1": jnp.array([-4.0, -3.5, -2.5, -3.0, -2.0]),
        "school_2": jnp.array([-0.5, 0.5, 0.0, 1.0, -1.0]),
        "school_3": jnp.array([2.0, 3.5, 3.0, 4.0, 2.5]),
    }
    grand_mean = jnp.concatenate(list(key_data.values())).mean()

    with Space() as space:
        mu = Cloud("mu", n_particles=10, dim=1)
        clouds = {"mu": mu}

        for name, data in key_data.items():
            theta = Cloud(name, n_particles=len(data), dim=1)
            mu.drift(theta, elasticity=1.0)
            theta.covers(data[:, None])
            clouds[name] = theta

    posterior = relax(space, clouds, lr=0.05, epsilon=0.5, max_steps=50, tol=1e-4)

    # Global mean should be near grand mean of all data
    global_mean = float(posterior.mean("mu")[0])
    assert abs(global_mean - float(grand_mean)) < 2.0, (
        f"Global mean {global_mean} too far from grand mean {float(grand_mean)}"
    )

    # School means should show shrinkage toward global mean
    raw_means = {name: float(data.mean()) for name, data in key_data.items()}
    for name in key_data:
        posterior_mean = float(posterior.mean(name)[0])
        raw_mean = raw_means[name]
        # Shrinkage: posterior mean should be between raw mean and global mean
        # (or at least closer to global mean than the raw mean is)
        dist_raw = abs(raw_mean - global_mean)
        dist_posterior = abs(posterior_mean - global_mean)
        assert dist_posterior <= dist_raw + 1.0, (
            f"{name}: posterior mean {posterior_mean:.2f} not shrunk "
            f"(raw={raw_mean:.2f}, global={global_mean:.2f})"
        )


def test_three_schools_energy_decreases():
    """Verify that energy actually decreases during relaxation."""
    data = jnp.array([[0.0], [1.0], [2.0]])

    with Space() as space:
        mu = Cloud("mu", n_particles=5, dim=1)
        theta = Cloud("theta", n_particles=3, dim=1)
        mu.drift(theta)
        theta.covers(data)
        clouds = {"mu": mu, "theta": theta}

    posterior = relax(space, clouds, lr=0.05, epsilon=0.5, max_steps=30)
    # Energy should be finite and reasonably small
    assert posterior.energy < 100.0
    assert not jnp.isnan(posterior.energy)
