import os
import numpy as np


def read_text_files(data_dir):
    """Read all text files from a directory."""
    texts = []

    if not os.path.exists(data_dir):
        print(f"Warning: Data directory '{data_dir}' does not exist.")
        return texts

    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    return texts


def cross_entropy_loss(logits, targets):
    """Calculate cross entropy loss."""
    batch_size = targets.shape[0]

    # For numerical stability, subtract the maximum value
    log_probs = logits - np.max(logits, axis=1, keepdims=True)
    log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=1, keepdims=True))

    # Calculate loss
    loss = -np.sum(log_probs[np.arange(batch_size), targets]) / batch_size
    return loss


def softmax(x):
    """Apply softmax function."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """Apply top-k and top-p (nucleus) filtering to logits."""
    # Apply top-k filtering
    if top_k > 0:
        indices_to_remove = logits < np.partition(logits, -top_k)[-top_k]
        logits[indices_to_remove] = filter_value

    # Apply top-p (nucleus) filtering
    if top_p > 0.0:
        sorted_logits = np.sort(logits)[::-1]
        sorted_indices = np.argsort(logits)[::-1]
        cumulative_probs = np.cumsum(softmax(sorted_logits))

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
        sorted_indices_to_remove[0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits