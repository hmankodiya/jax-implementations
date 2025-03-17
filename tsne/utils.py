import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def multimodal_random_generator(key, shape, n_modes=3, minval=1.0, maxval=20.0):
    """
    Generate a multivariate dataset with multiple modalities using a mixture of Gaussians.

    Args:
        key (jax.random.PRNGKey): Random key for reproducibility.
        shape (tuple): Desired output shape (batch_size, features).
        n_modes (int): Number of Gaussian clusters.
        minval (float): Minimum value for means.
        maxval (float): Maximum value for means.

    Returns:
        jnp.ndarray: Multimodal random samples.
        jnp.ndarray: Cluster indices (which Gaussian cluster each sample belongs to).
    """
    key, subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 5)

    batch_size, features = shape

    # Generate random cluster means
    means = jax.random.uniform(
        subkey1, shape=(n_modes, features), minval=minval, maxval=maxval
    )

    # Generate random cluster scales (diagonal covariance)
    scales = jax.random.uniform(
        subkey2, shape=(n_modes, features), minval=0.5, maxval=3.0
    )

    # Assign each sample to a cluster
    cluster_assignments = jax.random.randint(
        subkey3, shape=(batch_size,), minval=0, maxval=n_modes
    )

    # Generate noise for all samples
    noise = jax.random.normal(subkey4, shape=(batch_size, features))

    # Use JAX indexing to vectorize sample generation
    selected_means = means[cluster_assignments]
    selected_scales = scales[cluster_assignments]

    # Compute final samples
    samples = selected_means + selected_scales * noise

    return samples, cluster_assignments


def plot_features(data, clusters=None, title="Feature Projection (PCA/t-SNE)"):
    """
    Plots a scatter plot of 2D feature representations (PCA or t-SNE).
    If clusters are provided, points are colored based on cluster assignments.

    Parameters:
    - data (numpy.ndarray or jax.numpy.ndarray): 2D array with projected features.
    - clusters (numpy.ndarray or jax.numpy.ndarray, optional): Cluster assignments for each data point.
    - title (str): Title of the plot (default: "Feature Projection (PCA/t-SNE)").
    """
    plt.figure(figsize=(8, 6))

    if clusters is not None:
        scatter = plt.scatter(
            data[:, 0], data[:, 1], c=clusters, cmap="viridis", alpha=0.7
        )
        plt.colorbar(scatter, label="Cluster Index")
    else:
        plt.scatter(data[:, 0], data[:, 1], alpha=0.7, color="blue")

    plt.xlabel("Axis 1")
    plt.ylabel("Axis 2")
    plt.title(title)
    plt.grid(True)

    plt.show()
