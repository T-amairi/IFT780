import numpy as np


class Flatten:
    def forward(self, X, **kwargs):
        self.cache = X.shape
        N = X.shape[0]

        return X.reshape(N, -1)

    def backward(self, dA, **kwargs):
        return dA.reshape(self.cache)

    def get_params(self):
        return {}

    def get_gradients(self):
        return {}

    def reset(self):
        pass
