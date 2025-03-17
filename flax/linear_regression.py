import flax
import flax.nnx as nnx
from matplotlib import pyplot as plt
import optax
import jax
import jax.numpy as jnp

from tqdm import tqdm


# Define the model
class SimpleLinearRegression(nnx.Module):
    def __init__(self, in_features, out_features, rngs):
        self.linear = nnx.Linear(
            in_features=in_features, out_features=out_features, rngs=rngs
        )

    def __call__(self, x):
        return self.linear(x)


def loss_fn(model, batch):
    x, y = batch
    logits = model(x)
    return jnp.mean((logits - y) ** 2)


@nnx.jit
def train_step(model, optimizer, batch):
    # Compute the loss and gradients
    grad_fn = nnx.value_and_grad(loss_fn, argnums=0)
    loss, grads = grad_fn(model, batch)

    # Update the parameters using the gradients
    optimizer.update(grads)

    return loss


def train_loop(model, dataloader, epochs=100):
    adam = optax.adam(learning_rate=0.005)
    optimizer = nnx.Optimizer(model, adam)
    iterator = tqdm(range(epochs))
    loss = []

    for epoch_i in iterator:
        for step_i, batch in enumerate(dataloader):
            loss_i = train_step(model, optimizer, batch)

        loss.append(loss_i)

        iterator.set_postfix({"loss": loss_i})

    return loss


@nnx.jit
def inference(model, x):
    return jax.vmap(model)(x)


if __name__ == "__init__":
    key = jax.random.PRNGKey(42)
    dataset_size = 100
    spread = 0
    dataloader = [
        [
            jnp.arange(1, dataset_size + 1).reshape(-1, 1),
            (
                jnp.arange(1, dataset_size + 1)
                + jax.random.normal(key, dataset_size) * spread
            ).reshape(-1, 1),
        ],
    ]

    model = SimpleLinearRegression(1, 1, rngs=nnx.Rngs(42))

    loss = train_loop(model, dataloader, epochs=100)
    plt.plot(loss)
