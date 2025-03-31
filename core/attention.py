import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np

from core.utils import to_gpu, to_cpu


class SelfAttention:
    def __init__(self, embed_size, heads):
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.W_Q = to_gpu(np.random.randn(embed_size, embed_size) * 0.01)
        self.W_K = to_gpu(np.random.randn(embed_size, embed_size) * 0.01)
        self.W_V = to_gpu(np.random.randn(embed_size, embed_size) * 0.01)
        self.W_O = to_gpu(np.random.randn(embed_size, embed_size) * 0.01)

        self.grad_W_Q = to_gpu(np.zeros_like(self.W_Q))
        self.grad_W_K = to_gpu(np.zeros_like(self.W_K))
        self.grad_W_V = to_gpu(np.zeros_like(self.W_V))
        self.grad_W_O = to_gpu(np.zeros_like(self.W_O))

        self.cache = None

    def forward(self, inputs, mask=None):
        batch_size, seq_len, _ = inputs.shape

        Q = np.dot(inputs, self.W_Q)  # (batch_size, seq_len, embed_size)
        K = np.dot(inputs, self.W_K)  # (batch_size, seq_len, embed_size)
        V = np.dot(inputs, self.W_V)  # (batch_size, seq_len, embed_size)

        Q = Q.reshape(batch_size, seq_len, self.heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.heads, self.head_dim)

        Q = Q.transpose(0, 2, 1, 3)  # (batch_size, heads, seq_len, head_dim)
        K = K.transpose(0, 2, 1, 3)  # (batch_size, heads, seq_len, head_dim)
        V = V.transpose(0, 2, 1, 3)  # (batch_size, heads, seq_len, head_dim)

        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask

        attention_weights = self.softmax(scores)

        output = np.matmul(attention_weights, V)  # (batch_size, heads, seq_len, head_dim)

        output = output.transpose(0, 2, 1, 3)  # (batch_size, seq_len, heads, head_dim)
        output = output.reshape(batch_size, seq_len, self.embed_size)

        output = np.dot(output, self.W_O)

        self.cache = (inputs, Q, K, V, attention_weights)
        return output

    def backward(self, d_output):
        inputs, Q, K, V, attention_weights = self.cache
        batch_size, seq_len, _ = inputs.shape

        # Gradient dla W_O
        d_output_reshaped = d_output.copy()
        self.grad_W_O += np.dot(d_output_reshaped.reshape(-1, self.embed_size).T,
                                d_output.reshape(-1, self.embed_size))

        # Gradient dla warstw wielogłowicowych
        d_multihead = np.dot(d_output, self.W_O.T)
        d_multihead = d_multihead.reshape(batch_size, seq_len, self.heads, self.head_dim)
        d_multihead = d_multihead.transpose(0, 2, 1, 3)

        # Gradient dla V
        d_V = np.matmul(attention_weights.transpose(0, 1, 3, 2), d_multihead)
        d_V = d_V.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_size)
        self.grad_W_V += np.dot(inputs.reshape(-1, self.embed_size).T,
                                d_V.reshape(-1, self.embed_size))

        # Gradient dla wag uwagi
        d_attention = np.matmul(d_multihead, V.transpose(0, 1, 3, 2))
        d_attention_weights = d_attention * attention_weights * (1 - attention_weights)

        # Gradient dla K
        d_K = np.matmul(d_attention_weights.transpose(0, 1, 3, 2), Q)
        d_K = d_K.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_size)
        self.grad_W_K += np.dot(inputs.reshape(-1, self.embed_size).T,
                                d_K.reshape(-1, self.embed_size))

        # Gradient dla Q
        d_Q = np.matmul(d_attention_weights, K)
        d_Q = d_Q.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_size)
        self.grad_W_Q += np.dot(inputs.reshape(-1, self.embed_size).T,
                                d_Q.reshape(-1, self.embed_size))

        d_inputs = np.dot(d_Q, self.W_Q.T) + np.dot(d_K, self.W_K.T) + np.dot(d_V, self.W_V.T)

        return d_inputs

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)