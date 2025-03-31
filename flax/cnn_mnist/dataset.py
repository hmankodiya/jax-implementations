import random
import numpy as np
import torch
from torchvision import datasets, transforms
import jax
import jax.numpy as jnp


def load_mnist(split="train"):
    # Define transformation: normalize and convert to tensor
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Converts to (C, H, W) with values in [0, 1]
        ]
    )

    # Load the dataset
    dataset = datasets.MNIST(
        root="./data", train=(split == "train"), download=True, transform=transform
    )

    # Stack all data and targets into a single tensor
    data = torch.stack([img for img, _ in dataset])  # Shape: [N, 1, 28, 28]
    targets = torch.tensor([label for _, label in dataset])  # Shape: [N]

    return data, targets


class DataLoader:
    def __init__(
        self,
        data,
        targets=None,
        return_labels=False,
        n_device=None,
        batch_size=1,
        shuffle=False,
    ):
        self.data = data.numpy()
        self.targets = None
        if targets is not None:
            self.targets = targets.numpy()
        self.return_labels = return_labels

        self.batch_size = batch_size
        self.indices = list(range(len(data)))
        self.n_device = n_device

        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.indices)

    def transform(self, x, use_targets=False):
        if use_targets:
            if not self.return_labels:
                one_hot_targets = np.zeros((len(x), 10), dtype=np.float32)
                one_hot_targets[np.arange(len(x)), x] = 1.0
                return one_hot_targets
            return x

        # Convert [B, 1, 28, 28] → [B, 28, 28, 1] (PyTorch → JAX)
        x = np.transpose(x, (0, 2, 3, 1))
        return x.astype(np.float32)

    def collate(self):
        batch_indices = self.indices[
            self.current_index : self.current_index + self.batch_size
        ]
        self.current_index += self.batch_size

        batch_images = self.transform(self.data[batch_indices])
        if self.targets is not None:
            batch_targets = self.transform(
                self.targets[batch_indices], use_targets=True
            )

        batch = (jnp.array(batch_images),)
        if self.targets is not None:
            batch += (jnp.array(batch_targets),)

        return batch

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= len(self.indices):
            raise StopIteration
        return self.collate()

    def __len__(self):
        return len(self.data)
