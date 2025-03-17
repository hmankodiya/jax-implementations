from typing import NamedTuple

import jax
import jax.numpy as jnp


class PCAState(NamedTuple):
    components: jax.Array
    means: jax.Array
    explained_variance: jax.Array


def transform(state, x):
    x = x - state.means
    return jnp.dot(x, jnp.transpose(state.components))


def reconstruct(state, x):
    return jnp.dot(x, state.components) + state.means


def fit(x, n_components):
    n_samples, n_features = x.shape

    means = x.mean(axis=0, keepdims=True)
    x = x - means

    # Decomposing matrix in sub-matrices
    U, S, Vt = jax.scipy.linalg.svd(x, full_matrices=False)

    # Compute the explained variance
    explained_variance = (S[:n_components] ** 2) / (n_samples - 1)

    # Return the transformation matrix
    P = Vt[:n_components]

    return PCAState(components=P, means=means, explained_variance=explained_variance)

def fit_transform(x, n_components):
    pca_state = fit(x, n_components)
    return transform(pca_state, x)