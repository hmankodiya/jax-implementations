from tqdm import tqdm

import optax
import jax
import jax.numpy as jnp
from flax import nnx  # The Flax NNX API.
from functools import partial


class Classifier(nnx.Module):
    """A simple CNN model."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
        self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, 10, rngs=rngs)

    def __call__(self, x):
        x = self.avg_pool(nnx.relu(self.conv1(x)))
        x = self.avg_pool(nnx.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nnx.relu(self.linear1(x))
        x = self.linear2(x)
        return x


def compute_loss(model, batch):
    x, y = batch
    logits = model(x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    return loss


@nnx.jit
def train_step(model, optimizer, batch):
    loss_fn = lambda m: compute_loss(m, batch)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss


@nnx.jit
def eval_step(model, batch):
    x, y = batch
    logits = model(x)
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == y)
    return accuracy


def train(model, dataloader, epochs=5, lr=1e-3):
    optimizer_def = optax.adam(lr)
    optimizer = nnx.Optimizer(model, optimizer_def)
    losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}", total=64):
            loss = train_step(model, optimizer, batch)
            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / num_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

    return losses
