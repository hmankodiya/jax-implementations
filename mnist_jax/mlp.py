import os
import tqdm

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


class MLP:
    def __init__(self, seed=42, input_dims=784, num_classes=10) -> None:
        self.key = jax.random.PRNGKey(seed)
        self.input_dims = input_dims
        self.num_classes = num_classes
        self.params = {"weights": [], "bias": []}

    def init_layer(self, layer_widths, scale=0.01):
        layer_widths = [self.input_dims] + layer_widths + [self.num_classes]
        keys = jax.random.split(self.key, num=len(layer_widths) - 1)
        for i, key in enumerate(keys):
            weight_key, bias_key = jax.random.split(key)
            self.params["weights"].append(
                scale
                * jax.random.normal(weight_key, (layer_widths[i], layer_widths[i + 1]))
            )
            self.params["bias"].append(
                scale * jax.random.normal(bias_key, shape=(layer_widths[i + 1],))
            )

        return self.params

    @staticmethod
    def _forward(layer_activations, layer_param):
        w, b = layer_param
        activations = jax.nn.relu(jnp.dot(layer_activations, w) + b)
        return activations

    @staticmethod
    def forward(params, x):
        activations = x
        for w, b in zip(params["weights"], params["bias"]):
            activations = jax.vmap(MLP._forward, in_axes=(0, None))(activations, [w, b])

        return activations - jax.nn.logsumexp(activations, axis=1)[:, None]


def CCE(params, x, y):
    logits = MLP.forward(params, x)
    return -jnp.mean(y * logits)


def accuracy(pred_classes, y):
    return jnp.mean(pred_classes == y)


@jax.jit
def update(params, x, y, lr=0.01):
    loss, grads = jax.value_and_grad(CCE)(params, x, y)
    return loss, jax.tree.map(lambda p, g: p - lr * g, params, grads)


def train(mlp, dataloader, epochs):
    params = mlp.params
    epoch_iterator = tqdm.tqdm(
        range(epochs), desc="Epoch", position=0, total=epochs, dynamic_ncols=True
    )

    loss_epoch = []
    for epoch in epoch_iterator:
        loss_i = []
        for x, y in dataloader:
            loss, params = update(params, x, y)
            loss_i.append(float(loss))

        mean_epoch_loss = sum(loss_i) / len(loss_i)
        loss_epoch.append(float(mean_epoch_loss))
        epoch_iterator.set_postfix({"train_loss": float(mean_epoch_loss)})

    mlp.params = params
    return loss_epoch


def validate(mlp, dataloader):
    print('Validating')
    val_accuracy = []
    for x, y in dataloader:
        _, pred_labels = inference(mlp, x)
        accuracy_i = accuracy(pred_labels, y)
        val_accuracy.append(float(accuracy_i))

    val_accuracy = sum(val_accuracy) / len(val_accuracy)
    return val_accuracy


def inference(mlp, x):
    params = mlp.params
    outs = jnp.exp(MLP.forward(params, x))
    labels = jnp.argmax(outs, axis=1)
    return outs, labels
