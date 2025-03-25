from functools import partial
import jax
import jax.numpy as jnp


class KMeans:
    def __init__(self, x, n_clusters=8, init="auto", max_iter=300, tol=1e-4, seed=42):
        self.key = jax.random.PRNGKey(seed=seed)
        self.n_samples, self.input_dims = x.shape
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol

        self.x = x
        self.centroids = self.init_centroids()

    def init_centroids(self):
        if self.init == "kmeans++":
            NotImplementedError("kmeans++ initialization strat not implemented")
        elif self.init == "auto":
            indices = jnp.arange(self.n_samples)
            random_indices = jax.random.choice(
                self.key, indices, shape=(self.n_clusters,), replace=False
            )
            centroids = self.x[random_indices]
            return centroids
        else:
            raise ValueError(f"{self.init_strategy} not a valid strategy")

    @staticmethod
    @jax.jit
    def compute_norm(x, y):
        x_i = x[:, None, :]  # Shape (N, 1, D)
        y_j = y[None, :, :]  # Shape (1, K, D)
        distances = jnp.sum((x_i - y_j) ** 2, axis=-1)  # Squared Euclidean
        return distances

    @staticmethod
    @partial(jax.jit)
    def assign_centroids(x, centroids):
        distances = KMeans.compute_norm(x, centroids)
        assignments = distances.argmin(axis=-1)
        return assignments, distances

    # new_centroids = centroids  # jnp.zeros((n_clusters, input_dims))

    @staticmethod
    @partial(jax.jit, static_argnums=(2,))
    def update_centroids(x, assignments, n_clusters):
        def update_centroid(i):
            mask = assignments == i  # shape: (n,)
            cluster_points = jnp.where(mask[:, None], x, 0.0)
            return cluster_points.sum(axis=0) / mask.sum(axis=0)

        new_centroids = jax.vmap(update_centroid)(jnp.arange(n_clusters))
        return new_centroids

    def fit(self, verbose=False, track_history=False):
        centroids = self.centroids
        centroids_history = [] if track_history else None
        assignment_history = [] if track_history else None
        
        for i in range(self.max_iter):
            assignments, distances = KMeans.assign_centroids(self.x, centroids)
            centroids = KMeans.update_centroids(self.x, assignments, self.n_clusters)
            # print(centroids, assignments)
            if verbose and i % 10 == 0:
                print(f"Epoch {i}: distance = {distances.mean():.4f}")

            if track_history:
                assignment_history.append(assignments)
                centroids_history.append(centroids)

        self.centroids = centroids
        self.assignments = assignments

        if track_history:
            return centroids, assignments, distances.mean(), centroids_history, assignment_history
        else:
            return centroids, assignments, distances.mean()

    def predict(self, x):
        distances = self.compute_norm(x, self.centroids)
        return distances.argmin(axis=1)
