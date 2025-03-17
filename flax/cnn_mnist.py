from matplotlib import pyplot as plt

import flax
import flax.nnx as nnx
import optax
import jax
import jax.numpy as jnp

from tqdm import tqdm


class Classififer(nnx.Module):
    def __init__(self, rngs, num_classes=10):
        self.nums_classes = 10
        self.conv1 = nnx.Conv(
            in_features=3,
            out_features=96,
            kernel_size=(11, 11),
            strides=(4, 4),
            padding="VALID",
            kernel_init=nnx.nn.initializers.he_normal(),
        )

        # max: pool_size=(3,3), strides= (2,2), padding= 'valid', data_format= None)

        self.conv2 = nnx.Conv(
            in_features=96,
            out_features=256,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="SAME",
            kernel_init=nnx.nn.initializers.he_normal(),
        )

        self.conv3 = nnx.Conv(
            in_features=256,
            out_features=384,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            kernel_init=nnx.nn.initializers.he_normal(),
        )

        self.conv4 = nnx.Conv(
            in_features=384,
            out_features=384,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            kernel_init=nnx.nn.initializers.he_normal(),
        )
        self.conv5 = nnx.Conv(
            in_features=384,
            out_features=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            kernel_init=nnx.nn.initializers.he_normal(),
        )

        self.fc1 = nnx.Dense(4096, kernel_init=nnx.nn.initializers.he_normal())
        self.fc2 = nnx.Dense(4096, kernel_init=nnx.nn.initializers.he_normal())
        self.fc3 = nnx.Dense(1000, kernel_init=nnx.nn.initializers.he_normal())
        self.fc4 = nnx.Dense(num_classes, kernel_init=nnx.nn.initializers.he_normal())

    def __call__(self, x):
        # 1st Convolutional Layer
        x = self.conv1(x)
        x = nnx.relu(x)
        x = nnx.nn.max_pool(x, pool_size=(3, 3), strides=(2, 2), padding="VALID")

        # 2nd Convolutional Layer
        x = self.conv2(x)
        x = nnx.relu(x)
        x = nnx.nn.max_pool(x, pool_size=(3, 3), strides=(2, 2), padding="VALID")

        # 3rd Convolutional Layer
        x = self.conv3(x)
        x = nnx.relu(x)

        # 4th Convolutional Layer
        x = self.conv4(x)
        x = nnx.relu(x)

        # 5th Convolutional Layer
        x = self.conv5(x)
        x = nnx.relu(x)
        x = nnx.nn.max_pool(x, pool_size=(3, 3), strides=(2, 2), padding="VALID")

        # Flattening the output
        x = x.reshape((x.shape[0], -1))  # Flatten

        # Fully Connected Layers
        x = self.fc1(x)
        x = nnx.relu(x)
        x = self.fc2(x)
        x = nnx.relu(x)
        x = self.fc3(x)
        x = nnx.relu(x)

        # Output Layer
        x = self.fc4(x)

        return x

def loss_fn(model, batch):
    x, y = batch
    logits = model(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y)
    
  
@nnx.jit      
def train_step(model, optimizer, batch):
    grad_fn = nnx.value_and_grad(loss_fn, argnums=0)
    loss, grads = grad_fn(model, batch)
    
    optimizer.update(grads)
    
    return loss

@nnx.jit
def inference(model, x):
    probabilities = jax.vmap(nnx.softmax)
    return jax.vamp(model)(x)

def train_loop(model, dataloader, epochs=100):
    adam = optax.adam(learning_rate=0.005)
    optimizer = nnx.Optimizer(model, adam)
    iterator = tqdm(range(epochs))
    loss = []
    
    for epoch_i in iterator:
        for step_i, batch in enumerate(dataloader):
            loss_i = train_step(model, optimizer, batch)
            
        loss.append(loss_i)
        
        iterator.set_postfix({'loss': loss_i})
        
    return loss