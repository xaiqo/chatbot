import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np

from config import USE_GPU
from core.utils import to_gpu, to_cpu


class Dropout:
    def __init__(self, rate=0.1):
        self.rate = rate
        self.mask = None

    def forward(self, x, training=True):
        if not training:
            return x

        if USE_GPU and CUPY_AVAILABLE:
            self.mask = cp.random.rand(*x.shape) > self.rate
        else:
            self.mask = np.random.rand(*x.shape) > self.rate

        return x * self.mask / (1 - self.rate)

    def backward(self, d_out):
        return d_out * self.mask / (1 - self.rate)


class LayerNorm:
    def __init__(self, features, eps=1e-6):
        self.gamma = to_gpu(np.ones(features))
        self.beta = to_gpu(np.zeros(features))
        self.eps = eps
        self.cache = None
        self.grad_gamma = None
        self.grad_beta = None

    def forward(self, x):
        self.cache = x

        if USE_GPU and CUPY_AVAILABLE:
            mean = cp.mean(x, axis=-1, keepdims=True)
            var = cp.var(x, axis=-1, keepdims=True)
        else:
            mean = np.mean(x, axis=-1, keepdims=True)
            var = np.var(x, axis=-1, keepdims=True)

        std = np.sqrt(var + self.eps)
        normalized = (x - mean) / std

        return self.gamma * normalized + self.beta

    def backward(self, d_out):
        x = self.cache

        if USE_GPU and CUPY_AVAILABLE:
            mean = cp.mean(x, axis=-1, keepdims=True)
            var = cp.var(x, axis=-1, keepdims=True)
        else:
            mean = np.mean(x, axis=-1, keepdims=True)
            var = np.var(x, axis=-1, keepdims=True)

        std = np.sqrt(var + self.eps)
        normalized = (x - mean) / std

        batch_size = x.shape[0]

        self.grad_gamma = np.sum(d_out * normalized, axis=(0, 1), keepdims=True)
        self.grad_beta = np.sum(d_out, axis=(0, 1), keepdims=True)

        d_normalized = d_out * self.gamma
        dx_norm = d_normalized / std
        dvar = np.sum(d_normalized * (x - mean) * -0.5 * (var + self.eps) ** (-1.5), axis=-1, keepdims=True)
        dmean = np.sum(-d_normalized / std, axis=-1, keepdims=True) + dvar * np.mean(-2.0 * (x - mean), axis=-1,
                                                                                     keepdims=True)

        dx = d_normalized / std
        dx += dvar * 2.0 * (x - mean) / batch_size
        dx += dmean / batch_size

        return dx


class FeedForward:
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1):
        self.W1 = to_gpu(np.random.randn(input_dim, hidden_dim) * 0.01)
        self.b1 = to_gpu(np.zeros(hidden_dim))
        self.W2 = to_gpu(np.random.randn(hidden_dim, input_dim) * 0.01)
        self.b2 = to_gpu(np.zeros(input_dim))
        self.dropout = Dropout(dropout_rate)
        self.cache = None
        self.grad_W1 = to_gpu(np.zeros_like(self.W1))
        self.grad_b1 = to_gpu(np.zeros_like(self.b1))
        self.grad_W2 = to_gpu(np.zeros_like(self.W2))
        self.grad_b2 = to_gpu(np.zeros_like(self.b2))

    def forward(self, x, training=True):
        h = np.dot(x, self.W1) + self.b1
        h_relu = np.maximum(0, h)
        h_dropout = self.dropout.forward(h_relu, training)
        out = np.dot(h_dropout, self.W2) + self.b2
        self.cache = (x, h, h_relu, h_dropout)
        return out

    def backward(self, d_out):
        x, h, h_relu, h_dropout = self.cache

        d_h_dropout = np.dot(d_out, self.W2.T)
        d_h_relu = self.dropout.backward(d_h_dropout)
        d_h = d_h_relu * (h > 0)

        self.grad_W2 += np.dot(h_dropout.T, d_out)
        self.grad_b2 += np.sum(d_out, axis=0)
        self.grad_W1 += np.dot(x.T, d_h)
        self.grad_b1 += np.sum(d_h, axis=0)

        d_x = np.dot(d_h, self.W1.T)
        return d_x
