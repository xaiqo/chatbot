import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np

from config import USE_GPU
from  core.utils import to_gpu, to_cpu


class EmbeddingLayer:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = to_gpu(np.random.randn(vocab_size, embedding_dim) * 0.01)
        self.grad = to_gpu(np.zeros_like(self.embeddings))
        self.input_tokens = None

    def forward(self, input_tokens):
        self.input_tokens = input_tokens

        if USE_GPU and CUPY_AVAILABLE:
            batch_embeddings = cp.take(self.embeddings, input_tokens, axis=0)
        else:
            batch_embeddings = np.take(self.embeddings, input_tokens, axis=0)

        return batch_embeddings

    def backward(self, d_out):
        if USE_GPU and CUPY_AVAILABLE:
            for i in range(len(self.input_tokens)):
                cp.add.at(self.grad, self.input_tokens[i], d_out[i])
        else:
            for i in range(len(self.input_tokens)):
                np.add.at(self.grad, self.input_tokens[i], d_out[i])

        return None
