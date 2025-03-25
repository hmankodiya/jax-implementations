import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def multimodal_random_generator(
    key, shape, n_modes=3, minval=1.0, maxval=20.0, separation_factor=1.0
):
    """
    Generate a multivariate dataset with multiple modalities using a mixture of Gaussians.
    Allows control over cluster separation.

    Args:
        key (jax.random.PRNGKey): Random key for reproducibility.
        shape (tuple): Desired output shape (batch_size, features).
        n_modes (int): Number of Gaussian clusters.
        minval (float): Minimum base value for means.
        maxval (float): Maximum base value for means.
        separation_factor (float): Controls how far apart cluster centers are.

    Returns:
        jnp.ndarray: Multimodal random samples.
        jnp.ndarray: Cluster indices (which Gaussian cluster each sample belongs to).
    """
    key, subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 5)

    batch_size, features = shape

    # Generate normalized random directions for means
    raw_directions = jax.random.normal(subkey1, shape=(n_modes, features))
    norms = jnp.linalg.norm(raw_directions, axis=1, keepdims=True) + 1e-8
    unit_directions = raw_directions / norms

    # Spread means out along those directions with separation_factor
    mean_magnitudes = jax.random.uniform(
        subkey2, shape=(n_modes, 1), minval=minval, maxval=maxval
    )
    means = unit_directions * mean_magnitudes * separation_factor

    # Random scales for clusters
    scales = jax.random.uniform(
        subkey3, shape=(n_modes, features), minval=0.5, maxval=3.0
    )

    # Assign clusters to each sample
    cluster_assignments = jax.random.randint(
        subkey4, shape=(batch_size,), minval=0, maxval=n_modes
    )

    # Generate noise
    key, subkey5 = jax.random.split(key)
    noise = jax.random.normal(subkey5, shape=(batch_size, features))

    selected_means = means[cluster_assignments]
    selected_scales = scales[cluster_assignments]

    samples = selected_means + selected_scales * noise

    return samples, cluster_assignments


def plot_features(data, clusters=None, centroids=None, title="Clusters"):
    plt.figure(figsize=(8, 6))

    if clusters is not None:
        scatter = plt.scatter(
            data[:, 0], data[:, 1], c=clusters, cmap="viridis", alpha=0.7
        )
        plt.colorbar(scatter, label="Cluster Index")
    else:
        plt.scatter(data[:, 0], data[:, 1], alpha=0.7, color="blue")

    # Plot centroids if provided
    if centroids is not None:
        plt.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="X",
            s=200,
            c="red",
            edgecolors="black",
            linewidths=1.5,
            label="Centroids",
        )
        plt.legend()

    plt.xlabel("Axis 1")
    plt.ylabel("Axis 2")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_centroid_trajectory(centroids_history, data=None, clusters=None):
    """
    centroids_history: list of (k, 2) arrays from each iteration
    data: optional, the data points to show in the background
    clusters: optional, the cluster assignment of each data point
    """
    plt.figure(figsize=(8, 6))

    if data is not None:
        if clusters is not None:
            scatter = plt.scatter(
                data[:, 0], data[:, 1], c=clusters, cmap="viridis", alpha=0.5
            )
            plt.colorbar(scatter, label="Cluster Index")
        else:
            plt.scatter(data[:, 0], data[:, 1], alpha=0.5, color="gray")

    n_steps = len(centroids_history)
    centroids_history = jnp.stack(centroids_history)  # (n_steps, k, 2)
    k = centroids_history.shape[1]

    for cluster_idx in range(k):
        path = centroids_history[:, cluster_idx, :]
        plt.plot(
            path[:, 0],
            path[:, 1],
            marker="o",
            linestyle="--",
            label=f"Centroid {cluster_idx}",
        )

    plt.xlabel("Axis 1")
    plt.ylabel("Axis 2")
    plt.title("Centroid Trajectories Over Iterations")
    plt.grid(True)
    plt.legend()
    plt.show()
