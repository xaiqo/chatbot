import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np

from config import USE_GPU
from core.utils import to_gpu, to_cpu
from core.embedding import EmbeddingLayer
from core.transformer_block import TransformerBlock


class Transformer:
    def __init__(
            self,
            vocab_size,
            embed_size,
            num_blocks,
            heads,
            hidden_size,
            dropout_rate=0.1
    ):
        self.embedding = EmbeddingLayer(vocab_size, embed_size)
        self.blocks = [
            TransformerBlock(embed_size, heads, hidden_size, dropout_rate)
            for _ in range(num_blocks)
        ]
        self.W_out = to_gpu(np.random.randn(embed_size, vocab_size) * 0.01)
        self.b_out = to_gpu(np.zeros(vocab_size))
        self.grad_W_out = to_gpu(np.zeros_like(self.W_out))
        self.grad_b_out = to_gpu(np.zeros_like(self.b_out))
        self.cache = None
        self.parameters = {}
        self._build_parameters_dict()

    def _build_parameters_dict(self):
        self.parameters['embedding'] = self.embedding.embeddings
        self.parameters['embedding'].grad = self.embedding.grad

        self.parameters['W_out'] = self.W_out
        self.parameters['W_out'].grad = self.grad_W_out

        self.parameters['b_out'] = self.b_out
        self.parameters['b_out'].grad = self.grad_b_out

        for i, block in enumerate(self.blocks):
            self.parameters[f'block_{i}_W_Q'] = block.attention.W_Q
            self.parameters[f'block_{i}_W_Q'].grad = block.attention.grad_W_Q

            self.parameters[f'block_{i}_W_K'] = block.attention.W_K
            self.parameters[f'block_{i}_W_K'].grad = block.attention.grad_W_K

            self.parameters[f'block_{i}_W_V'] = block.attention.W_V
            self.parameters[f'block_{i}_W_V'].grad = block.attention.grad_W_V

            self.parameters[f'block_{i}_W_O'] = block.attention.W_O
            self.parameters[f'block_{i}_W_O'].grad = block.attention.grad_W_O

            # Parametry warstwy normalizacji
            self.parameters[f'block_{i}_norm1_gamma'] = block.norm1.gamma
            self.parameters[f'block_{i}_norm1_gamma'].grad = block.norm1.grad_gamma

            self.parameters[f'block_{i}_norm1_beta'] = block.norm1.beta
            self.parameters[f'block_{i}_norm1_beta'].grad = block.norm1.grad_beta

            self.parameters[f'block_{i}_norm2_gamma'] = block.norm2.gamma
            self.parameters[f'block_{i}_norm2_gamma'].grad = block.norm2.grad_gamma

            self.parameters[f'block_{i}_norm2_beta'] = block.norm2.beta
            self.parameters[f'block_{i}_norm2_beta'].grad = block.norm2.grad_beta

            # Parametry feed forward
            self.parameters[f'block_{i}_ff_W1'] = block.feed_forward.W1
            self.parameters[f'block_{i}_ff_W1'].grad = block.feed_forward.grad_W1

            self.parameters[f'block_{i}_ff_b1'] = block.feed_forward.b1
            self.parameters[f'block_{i}_ff_b1'].grad = block.feed_forward.grad_b1

            self.parameters[f'block_{i}_ff_W2'] = block.feed_forward.W2
            self.parameters[f'block_{i}_ff_W2'].grad = block.feed_forward.grad_W2

            self.parameters[f'block_{i}_ff_b2'] = block.feed_forward.b2
            self.parameters[f'block_{i}_ff_b2'].grad = block.feed_forward.grad_b2

    def forward(self, input_tokens, training=True):
        x = self.embedding.forward(input_tokens)

        for block in self.blocks:
            x = block.forward(x, training)

        logits = np.dot(x, self.W_out) + self.b_out

        self.cache = x
        return logits

    def backward(self, d_logits):
        x = self.cache

        self.grad_W_out += np.dot(x.T, d_logits)
        self.grad_b_out += np.sum(d_logits, axis=0)

        d_x = np.dot(d_logits, self.W_out.T)

        for block in reversed(self.blocks):
            d_x = block.backward(d_x)

        self.embedding.backward(d_x)

        return d_x

    def to_gpu(self):
        if not (USE_GPU and CUPY_AVAILABLE):
            print("GPU not installed")
            return

        for param_name, param in self.parameters.items():
            self.parameters[param_name] = to_gpu(param)
            self.parameters[param_name].grad = to_gpu(param.grad)

    def to_cpu(self):
        for param_name, param in self.parameters.items():
            self.parameters[param_name] = to_cpu(param)
            self.parameters[param_name].grad = to_cpu(param.grad)

    def save(self, filepath):
        self.to_cpu()

        parameters_dict = {}
        for param_name, param in self.parameters.items():
            parameters_dict[param_name] = to_cpu(param)

        np.save(filepath, parameters_dict)

    def load(self, filepath):
        parameters_dict = np.load(filepath, allow_pickle=True).item()

        for param_name, param in parameters_dict.items():
            if param_name in self.parameters:
                self.parameters[param_name] = to_gpu(param) if USE_GPU else param
