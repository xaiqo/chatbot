import numpy as np
from ..utils.helpers import cross_entropy_loss, softmax

class Trainer:
    def __init__(self, model, optimizer, dataset):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset

    def train(self, epochs=10, batch_size=32):
        for epoch in range(epochs):
            inputs, targets = self.dataset.get_batches(batch_size)
            total_loss = 0
            for batch_inputs, batch_targets in zip(inputs, targets):
                logits = self.model.forward(batch_inputs)
                loss = cross_entropy_loss(logits, batch_targets)
                total_loss += loss

                d_logits = self._compute_gradients(logits, batch_targets)
                self.model.backward(d_logits)

                self.optimizer.update(self.model)

            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(inputs):.4f}")

    def _compute_gradients(self, logits, targets):
        m = targets.shape[0]
        probs = softmax(logits)
        probs[range(m), targets] -= 1
        return probs / m