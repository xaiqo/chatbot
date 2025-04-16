import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np

from config import USE_GPU
from core.utils import to_gpu, to_cpu


class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0  # Timestep

    def update(self, model):
        self.t += 1
        lr_t = self.learning_rate * np.sqrt(1.0 - self.beta2 ** self.t) / (1.0 - self.beta1 ** self.t)

        for param_name, param_dict in model.parameters.items():
            # Initialize momentum and velocity if not present
            if param_name not in self.m:
                self.m[param_name] = np.zeros_like(param_dict['value'])
                self.v[param_name] = np.zeros_like(param_dict['value'])

            # Get parameter and its gradient
            param = param_dict['value']
            grad = param_dict['grad']

            # Add weight decay to gradient if specified
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param

            # Update biased first moment estimate
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad

            # Update biased second raw moment estimate
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad * grad)

            # Update parameters
            param_dict['value'] -= lr_t * self.m[param_name] / (np.sqrt(self.v[param_name]) + self.epsilon)

            # Reset gradients to zero
            param_dict['grad'] = np.zeros_like(param_dict['value'])