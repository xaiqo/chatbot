import os
import numpy as np
from config import (
    BPE_VOCAB_SIZE,
    BPE_MIN_FREQUENCY,
    EMBED_DIM,
    NUM_BLOCKS,
    HEADS,
    HIDDEN_SIZE,
    DROPOUT_RATE,
    USE_GPU,
    LEARNING_RATE,
    BETA1,
    BETA2,
    WEIGHT_DECAY,
    EPSILON,
    BATCH_SIZE,
    EPOCHS,
    MAX_SEQ_LENGTH,
    PROCESSED_DATA_DIR,
    MODEL_SAVE_PATH,
    TOKENIZER_SAVE_PATH
)
from core.transformer import Transformer
from training.tokenization.bpe import BPETokenizer
from training.optimizer import AdamOptimizer
from training.dataset import TextDataset
from utils.helpers import read_text_files, cross_entropy_loss, softmax


def main():
    """Main training function."""
    print("Starting model training...")

    # Check if processed data directory exists
    if not os.path.exists(PROCESSED_DATA_DIR):
        print(f"Processed data directory '{PROCESSED_DATA_DIR}' does not exist.")
        print("Please run preprocess_data.py first.")
        return

    # Read all processed text files
    print(f"Reading files from {PROCESSED_DATA_DIR}...")
    text_data = " ".join(read_text_files(PROCESSED_DATA_DIR))

    if not text_data:
        print("No text data found. Please make sure your PDF files are processed correctly.")
        return

    print(f"Loaded {len(text_data)} characters of text data.")

    # Initialize BPE tokenizer
    print("Training tokenizer...")
    tokenizer = BPETokenizer(vocab_size=BPE_VOCAB_SIZE)
    tokenizer.train(text_data, min_frequency=BPE_MIN_FREQUENCY)

    # Save tokenizer
    tokenizer.save(TOKENIZER_SAVE_PATH)
    print(f"Tokenizer saved to {TOKENIZER_SAVE_PATH}")

    # Prepare dataset
    print("Preparing dataset...")
    dataset = TextDataset(PROCESSED_DATA_DIR, tokenizer, seq_length=MAX_SEQ_LENGTH)

    # Initialize model
    print("Initializing model...")
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        embed_size=EMBED_DIM,
        num_blocks=NUM_BLOCKS,
        heads=HEADS,
        hidden_size=HIDDEN_SIZE,
        dropout_rate=DROPOUT_RATE
    )

    # Move model to GPU if available
    if USE_GPU:
        print("Using GPU for training.")
        model.to_gpu()

    # Initialize optimizer
    optimizer = AdamOptimizer(
        learning_rate=LEARNING_RATE,
        beta1=BETA1,
        beta2=BETA2,
        epsilon=EPSILON,
        weight_decay=WEIGHT_DECAY
    )

    # Training loop
    print(f"Starting training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        total_loss = 0
        batch_count = 0

        for batch_inputs, batch_targets in dataset.get_batches(batch_size=BATCH_SIZE):
            # Forward pass
            logits = model.forward(batch_inputs)

            # Calculate loss
            loss = 0
            for i in range(batch_targets.shape[1]):
                loss += cross_entropy_loss(logits[:, i, :], batch_targets[:, i])
            loss /= batch_targets.shape[1]

            total_loss += loss
            batch_count += 1

            # Backward pass and update
            d_logits = np.zeros_like(logits)
            for i in range(batch_targets.shape[1]):
                probs = softmax(logits[:, i, :])
                probs[np.arange(batch_targets.shape[0]), batch_targets[:, i]] -= 1
                d_logits[:, i, :] = probs / batch_targets.shape[0]

            model.backward(d_logits)
            optimizer.update(model)

            # Print progress
            if batch_count % 10 == 0:
                print(f"Epoch {epoch + 1}/{EPOCHS} | Batch {batch_count} | Loss: {loss:.4f}")

        # Print epoch summary
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch + 1}/{EPOCHS} completed | Average Loss: {avg_loss:.4f}")

        # Save model after each epoch
        model.save(f"{MODEL_SAVE_PATH}.epoch{epoch + 1}")
        print(f"Model saved to {MODEL_SAVE_PATH}.epoch{epoch + 1}")

    # Save final model
    model.save(MODEL_SAVE_PATH)
    print(f"Final model saved to {MODEL_SAVE_PATH}")
    print("Training completed.")


if __name__ == "__main__":
    main()