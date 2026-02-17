"""Cosine geometry on word embeddings.

A small set of hardcoded GloVe-style 10-d word vectors for common words.
With ``Space(geometry="cosine")``, the transport cost is based on
angular distance — so words pointing in similar directions are "close"
regardless of magnitude.

We set up two semantic clusters (animals vs vehicles) and a latent
cloud, then show the posterior reflects cosine-based grouping.
"""

import jax.numpy as jnp

from flowl import Cloud, Space, relax

# ── Hardcoded mini word vectors (GloVe-inspired, 10-d) ─────────────
# Semantically similar words point in similar directions.
EMBEDDINGS = {
    # Animals — share a common directional signature
    "cat":      [ 0.22,  0.05, -0.16,  0.31, -0.12,  0.09, -0.28,  0.17, -0.04,  0.11],
    "dog":      [ 0.24,  0.08, -0.14,  0.29, -0.10,  0.12, -0.25,  0.19, -0.06,  0.13],
    "horse":    [ 0.19,  0.03, -0.18,  0.33, -0.14,  0.07, -0.30,  0.15, -0.02,  0.09],
    "bird":     [ 0.20,  0.06, -0.12,  0.27, -0.08,  0.11, -0.22,  0.20, -0.07,  0.14],
    "fish":     [ 0.18,  0.04, -0.20,  0.25, -0.16,  0.05, -0.32,  0.13, -0.01,  0.07],
    "lion":     [ 0.21,  0.07, -0.15,  0.30, -0.11,  0.10, -0.27,  0.18, -0.05,  0.12],
    "tiger":    [ 0.23,  0.06, -0.17,  0.32, -0.13,  0.08, -0.29,  0.16, -0.03,  0.10],
    # Vehicles — a different directional signature
    "car":      [-0.15,  0.30,  0.22, -0.08,  0.25, -0.18,  0.14, -0.21,  0.33, -0.05],
    "truck":    [-0.13,  0.28,  0.20, -0.06,  0.23, -0.16,  0.12, -0.19,  0.31, -0.03],
    "bus":      [-0.17,  0.32,  0.24, -0.10,  0.27, -0.20,  0.16, -0.23,  0.35, -0.07],
    "train":    [-0.11,  0.26,  0.18, -0.04,  0.21, -0.14,  0.10, -0.17,  0.29, -0.01],
    "plane":    [-0.14,  0.29,  0.21, -0.07,  0.24, -0.17,  0.13, -0.20,  0.32, -0.04],
    "boat":     [-0.16,  0.31,  0.23, -0.09,  0.26, -0.19,  0.15, -0.22,  0.34, -0.06],
    "bicycle":  [-0.12,  0.27,  0.19, -0.05,  0.22, -0.15,  0.11, -0.18,  0.30, -0.02],
}

animals = ["cat", "dog", "horse", "bird", "fish", "lion", "tiger"]
vehicles = ["car", "truck", "bus", "train", "plane", "boat", "bicycle"]

animal_vecs = jnp.array([EMBEDDINGS[w] for w in animals], dtype=jnp.float32)
vehicle_vecs = jnp.array([EMBEDDINGS[w] for w in vehicles], dtype=jnp.float32)

print("Cosine Embeddings — Word Vector Clustering")
print("=" * 55)
print(f"  Animals:  {', '.join(animals)}")
print(f"  Vehicles: {', '.join(vehicles)}")
print()

# ── Cosine geometry ─────────────────────────────────────────────────
with Space(geometry="cosine") as space:
    mu = Cloud("mu", n_particles=20, dim=10)

    obs_animals = Cloud.from_samples("animals", animal_vecs, fixed=True)
    obs_animals.covers(animal_vecs)

    obs_vehicles = Cloud.from_samples("vehicles", vehicle_vecs, fixed=True)
    obs_vehicles.covers(vehicle_vecs)

    mu.drift(obs_animals, elasticity=1.0)
    mu.drift(obs_vehicles, elasticity=1.0)

    clouds = {"mu": mu, "animals": obs_animals, "vehicles": obs_vehicles}

posterior = relax(space, clouds, lr=0.005, epsilon=0.1, max_steps=200)

# ── Compare cosine similarities ────────────────────────────────────
mu_mean = posterior.mean("mu")
animal_centroid = jnp.mean(animal_vecs, axis=0)
vehicle_centroid = jnp.mean(vehicle_vecs, axis=0)


def cosine_sim(a, b):
    return float(jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))


print(f"Cosine similarity of posterior mean (mu) to each cluster:")
print(f"  mu · animals_centroid  = {cosine_sim(mu_mean, animal_centroid):+.4f}")
print(f"  mu · vehicles_centroid = {cosine_sim(mu_mean, vehicle_centroid):+.4f}")
print()
print(f"Final energy: {posterior.energy:.4f}")
print()
print("With cosine geometry, the latent cloud captures the angular midpoint")
print("between the two semantic clusters, not the Euclidean midpoint.")
