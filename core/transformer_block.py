import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np

from config import USE_GPU
from core.utils import to_gpu, to_cpu
from core.attention import SelfAttention
from core.layers import LayerNorm, Dropout, FeedForward


class TransformerBlock:
    def __init__(self, embed_size, heads, hidden_size, dropout_rate=0.1):
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = LayerNorm(embed_size)
        self.norm2 = LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, hidden_size, dropout_rate)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.cache = None

    def forward(self, x, training=True):
        attention_output = self.attention.forward(x)
        attention_output = self.dropout1.forward(attention_output, training)
        attention_output = x + attention_output  # Residual connection
        normalized1 = self.norm1.forward(attention_output)

        ff_output = self.feed_forward.forward(normalized1, training)
        ff_output = self.dropout2.forward(ff_output, training)
        output = normalized1 + ff_output  # Residual connection
        normalized2 = self.norm2.forward(output)

        self.cache = (x, attention_output, normalized1, ff_output, output)
        return normalized2

    def backward(self, d_out):
        x, attention_output, normalized1, ff_output, output = self.cache

        d_output = self.norm2.backward(d_out)

        d_ff_output = d_output
        d_normalized1 = d_output

        d_ff_output = self.dropout2.backward(d_ff_output)
        d_normalized1 += self.feed_forward.backward(d_ff_output)

        d_attention_output = self.norm1.backward(d_normalized1)

        d_attention = d_attention_output
        d_x = d_attention_output

        d_attention = self.dropout1.backward(d_attention)
        d_x += self.attention.backward(d_attention)

        return d_x