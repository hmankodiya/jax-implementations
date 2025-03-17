import flax
import flax.nnx as nnx
import optax

class SimpleLinearRegression(nnx.Module):
    def __init__(self, in_features, out_features, rngs):
        self.linear = nnx.Linear(in_features=in_features, out_features=out_features, rngs=rngs)

    def __call__(self, x):
        return self.linear(x)
    
if __name__ == '__init__':
    rngs = nnx.Rngs(42)
    model = SimpleLinearRegression(2, 1, rngs)
    nnx.display(model)


