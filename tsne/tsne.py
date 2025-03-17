from typing import NamedTuple

import jax
import jax.numpy as jnp

from pca import fit_transform


@jax.jit
def calculate_entropy(p):
    return jax.scipy.special.entr(p).sum(axis=1, keepdims=True)


@jax.jit
def calculate_perplexity(p):
    return 2 ** calculate_entropy(p)


class TSNE:
    def __init__(
        self,
        x,
        n_components,
        learning_rate=10.0,
        init_strategy="pca",
        seed=42,
        max_iter=1000,
        perplexity=30,
        tolerance=1.0e-6,
        lower_bound=0,
        upper_bound=1.0e4,
    ):
        self.key = jax.random.PRNGKey(seed=seed)
        self.n_samples, self.input_dims = x.shape
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.init_strategy = init_strategy

        self.perplexity = perplexity
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.x = x
        self.y = self.init_points()
        self.norm = TSNE.compute_norm(self.x)
        self.p, self.scaling_factor = self.compute_perplexity_scaling(
            norm=self.norm,
            perplexity=self.perplexity,
            tolerence=self.tolerance,
            max_iter=self.max_iter,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
        )

    def compute_perplexity_scaling(
        self,
        norm,
        perplexity,
        max_iter,
        tolerence=1.0e-6,
        lower_bound=0,
        upper_bound=1.0e4,
    ):

        n_samples = norm.shape[0]
        LB = jnp.full((n_samples, 1), lower_bound)
        UB = jnp.full((n_samples, 1), upper_bound)
        M = jnp.zeros((n_samples, 1))

        for _ in range(max_iter):
            M = (UB - LB) / 2 + LB
            scaling_factor = M
            p = TSNE.compute_p(norm, scaling_factor)
            current_perplexity = calculate_perplexity(p)

            mask0 = jnp.abs(current_perplexity - perplexity) > tolerence
            if (~mask0).all():
                break

            mask1 = (current_perplexity < perplexity) & mask0
            mask2 = (current_perplexity > perplexity) & mask0

            LB = jnp.where(mask1, M, LB)
            UB = jnp.where(mask2, M, UB)

        return p, M

    @staticmethod
    @jax.jit
    def compute_norm(x):
        x_i = x[None, :, :]  # Shape (1, N, D)
        x_j = x[:, None, :]  # Shape (N, 1, D)

        # Compute pairwise Euclidean distances
        norm = ((x_i - x_j) ** 2).sum(axis=-1)  # Shape (N, N)
        return norm

    @staticmethod
    @jax.jit
    def compute_p(norm, sig):
        # Compute affinity matrix (Gaussian kernel)
        norm_exp = jnp.exp(-norm / (2 * sig**2))
        p = norm_exp / norm_exp.sum(axis=-1, keepdims=True)

        return p

    @staticmethod
    @jax.jit
    def compute_q(y):
        y_i = y[None, :, :]
        y_j = y[:, None, :]
        distances = ((y_i - y_j) ** 2).sum(axis=-1)

        w_i_j = 1 / (1 + distances)
        z = w_i_j.sum(axis=-1, keepdims=True)
        q = w_i_j / z

        return q

    @staticmethod
    @jax.jit
    def kl_divergence(y, p):
        q = TSNE.compute_q(y)
        kl = jax.scipy.special.kl_div(p, q)
        return kl.sum()

    def init_points(self):
        if self.init_strategy == "pca":
            y = fit_transform(self.x, self.n_components)
            return y

        elif self.init_strategy == "random_init":
            self.key, sub_key1 = jax.random.split(self.key)
            y = jax.random.normal(sub_key1, shape=(self.n_samples, self.n_components))
            return y
        else:
            raise ValueError(f"{self.init_strategy} not a valid strategy")

    @staticmethod
    @jax.jit
    def update(y, p, learning_rate):
        kl_div_val, grad = jax.value_and_grad(TSNE.kl_divergence, argnums=0)(y, p)
        y = y - learning_rate * grad
        return kl_div_val, y, grad

    def fit(self, verbose=False):
        losses = []
        grad_norms = []
        y = self.y
        p = self.p

        for i in range(self.max_iter):
            kl_div_val, y, grad = self.update(y, p, self.learning_rate)
            grad_norm = jnp.linalg.norm(grad)  # Compute gradient norm

            losses.append(kl_div_val)
            grad_norms.append(grad_norm)

            if verbose and i % 10 == 0:
                print(
                    f"Epoch {i}: KL Divergence = {kl_div_val:.4f}, Grad Norm = {grad_norm:.4f}"
                )

        self.y = y
        outs = (losses, )
        if verbose:
            outs += (grad_norms, )
        
        return outs
