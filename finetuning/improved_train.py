#!/usr/bin/env python
"""
Improved training script for the Xaiqo chatbot
This script is designed to be run on Kaggle or other environments with GPU support
to fine-tune the GPT-2 model on custom data.
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    AdamW, 
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import numpy as np
import json
import logging

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class ConversationDataset(Dataset):
    """
    Dataset for training the chatbot on conversation data
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load data from file
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Process conversations
            for conversation in data:
                if isinstance(conversation, list):
                    # Format: list of message pairs
                    formatted_text = self.tokenizer.bos_token
                    for user_msg, bot_msg in conversation:
                        formatted_text += f"User: {user_msg}{self.tokenizer.sep_token}"
                        formatted_text += f"Bot: {bot_msg}{self.tokenizer.sep_token}"
                    
                    self.examples.append(formatted_text)
                elif isinstance(conversation, dict) and 'user' in conversation and 'bot' in conversation:
                    # Format: dict with 'user' and 'bot' keys
                    formatted_text = self.tokenizer.bos_token
                    formatted_text += f"User: {conversation['user']}{self.tokenizer.sep_token}"
                    formatted_text += f"Bot: {conversation['bot']}{self.tokenizer.sep_token}"
                    
                    self.examples.append(formatted_text)
        else:
            logger.warning(f"Data file {data_path} not found. Using empty dataset.")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        encodings = self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length")
        
        # Create labels (same as input_ids for language modeling)
        encodings["labels"] = encodings["input_ids"].copy()
        
        # Convert to tensors
        item = {key: torch.tensor(val) for key, val in encodings.items()}
        return item

def train(args):
    """
    Train the model on the provided dataset
    """
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Add special tokens
    special_tokens = {
        'pad_token': '<PAD>',
        'bos_token': '<BOS>',
        'eos_token': '<EOS>',
        'sep_token': '<SEP>'
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # Load base model
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))
    
    # Create dataset
    train_dataset = ConversationDataset(args.data_path, tokenizer, max_length=args.max_length)
    
    # Split into train and validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        warmup_steps=args.warmup_steps,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        save_total_limit=2,
        fp16=args.fp16,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model and tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info(f"Model saved to {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train the improved Xaiqo chatbot model")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the training data JSON file")
    parser.add_argument("--output-dir", type=str, default="./trained_model", help="Directory to save the model")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--eval-steps", type=int, default=500, help="Steps between evaluations")
    parser.add_argument("--save-steps", type=int, default=1000, help="Steps between saving checkpoints")
    parser.add_argument("--warmup-steps", type=int, default=200, help="Warmup steps for learning rate scheduler")
    parser.add_argument("--logging-steps", type=int, default=100, help="Steps between logging")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train the model
    train(args)

if __name__ == "__main__":
    main()