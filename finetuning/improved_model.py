#!/usr/bin/env python
"""
Improved Xaiqo Chatbot - All-in-One File

This file combines model definition, training, inference, and CLI interface
for the Xaiqo chatbot. It's designed to be easily shared and used on Kaggle
for training and inference.

Usage:
    # For training:
    python improved_model.py --train --data-path /kaggle/working/your_data.json --output-dir /kaggle/working/trained_model
    
    # For inference:
    python improved_model.py --message "Your message here" --model-path /kaggle/working/trained_model
"""

import os
import argparse
import sys
import json
import logging
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
import numpy as np

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

#######################
# MODEL IMPLEMENTATION #
#######################

class ImprovedChatbot:
    """
    An improved chatbot implementation that uses a pre-trained language model
    with better context handling and response generation capabilities.
    """
    def __init__(self, model_path=None, device=None):
        """
        Initialize the improved chatbot.
        
        Args:
            model_path: Path to a fine-tuned model (optional)
            device: Device to run the model on ('cuda' or 'cpu')
        """
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Add special tokens
        special_tokens = {
            'pad_token': '<PAD>',
            'bos_token': '<BOS>',
            'eos_token': '<EOS>',
            'sep_token': '<SEP>'
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Load base model or fine-tuned model
        if model_path and os.path.exists(model_path):
            print(f"Loading fine-tuned model from {model_path}")
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
        else:
            print("Loading base GPT-2 model")
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
            
        # Resize token embeddings to account for special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Conversation history for context management
        self.conversation_history = []
        self.max_history_length = 5  # Maximum number of turns to keep in history
        
    def _format_conversation(self):
        """
        Format the conversation history into a single string for the model.
        
        Returns:
            str: Formatted conversation history
        """
        formatted_text = self.tokenizer.bos_token
        
        for i, (user_msg, bot_msg) in enumerate(self.conversation_history):
            formatted_text += f"User: {user_msg}{self.tokenizer.sep_token}"
            if bot_msg:  # Bot message might be None for the latest user message
                formatted_text += f"Bot: {bot_msg}{self.tokenizer.sep_token}"
                
        return formatted_text
    
    def add_to_history(self, user_message, bot_response=None):
        """
        Add a message pair to the conversation history.
        
        Args:
            user_message: The user's message
            bot_response: The bot's response (can be None for the latest message)
        """
        self.conversation_history.append((user_message, bot_response))
        
        # Limit history length
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def generate_response(self, user_message, max_length=100, temperature=0.7, top_k=50, top_p=0.95):
        """
        Generate a response to the user's message.
        
        Args:
            user_message: The user's message
            max_length: Maximum length of the response
            temperature: Sampling temperature (higher = more random)
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability threshold for nucleus sampling
            
        Returns:
            str: Generated response
        """
        # Add user message to history (without bot response yet)
        self.add_to_history(user_message)
        
        # Format conversation history
        input_text = self._format_conversation() + "Bot:"
        
        # Encode the input text
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        
        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=input_ids.shape[1] + max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,  # Prevent repetition of 3-grams
                repetition_penalty=1.2,   # Penalize repetition
                num_return_sequences=1
            )
        
        # Decode the response
        response = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        # Clean up the response
        response = response.strip()
        if "User:" in response:
            response = response.split("User:")[0].strip()
            
        # Update history with the bot's response
        self.conversation_history[-1] = (user_message, response)
        
        return response
    
    def answer_question(self, question):
        """
        Process a user question and return an answer.
        This method is compatible with the original chatbot interface.
        
        Args:
            question: User's question
            
        Returns:
            str: Answer to the question
        """
        return self.generate_response(question)
    
    def clear_history(self):
        """
        Clear the conversation history.
        """
        self.conversation_history = []
    
    def save_config(self, filepath):
        """
        Save the chatbot configuration to a JSON file.
        
        Args:
            filepath: Path to save the configuration
        """
        config = {
            "max_history_length": self.max_history_length,
            "device": self.device
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_config(self, filepath):
        """
        Load the chatbot configuration from a JSON file.
        
        Args:
            filepath: Path to the configuration file
        """
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                config = json.load(f)
                
            self.max_history_length = config.get("max_history_length", 5)
            # Device is set during initialization, so we don't update it here

#######################
# TRAINING COMPONENTS #
#######################

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

def train_model(args):
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
    return model, tokenizer

#######################
# INFERENCE FUNCTIONS #
#######################

class ChatbotInference:
    def __init__(self, model_path=None):
        """
        Initialize the chatbot inference.

        Args:
            model_path: Path to the saved model
        """
        self.chatbot = ImprovedChatbot(model_path=model_path)

    def generate_response(self, question, max_length=100, temperature=0.7, top_k=50, top_p=0.95):
        """
        Generate a response for the given question.

        Args:
            question: User's question
            max_length: Maximum length of the response
            temperature: Sampling temperature (higher = more random)
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability threshold for nucleus sampling

        Returns:
            str: Generated response
        """
        return self.chatbot.generate_response(
            question,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

    def answer_question(self, question):
        """
        Process a user question and return an answer.

        Args:
            question: User's question

        Returns:
            str: Answer to the question
        """
        return self.chatbot.answer_question(question)

#######################
# CLI IMPLEMENTATION  #
#######################

def cli_interface(args):
    """
    Command-line interface for the chatbot
    """
    try:
        # Initialize the improved chatbot
        model_path = args.model_path if args.model_path else None
        device = args.device if args.device else None
        
        chatbot = ImprovedChatbot(model_path=model_path, device=device)
        
        # Clear history if requested
        if args.clear_history:
            chatbot.clear_history()
        
        # Get response
        response = chatbot.answer_question(args.message)
        
        # Print response (will be captured by Node.js)
        print(response)
        
        return 0
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1

#######################
# MAIN FUNCTION       #
#######################

def main():
    """
    Main function to handle both training and inference modes
    """
    parser = argparse.ArgumentParser(description="Xaiqo Improved Chatbot - All-in-One")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', help='Run in training mode')
    mode_group.add_argument('--message', type=str, help='Message to send to the chatbot (inference mode)')
    
    # Training arguments
    training_group = parser.add_argument_group('Training Arguments')
    training_group.add_argument("--data-path", type=str, help="Path to the training data JSON file")
    training_group.add_argument("--output-dir", type=str, default="/kaggle/working/trained_model", help="Directory to save the model")
    training_group.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    training_group.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    training_group.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    training_group.add_argument("--eval-steps", type=int, default=500, help="Steps between evaluations")
    training_group.add_argument("--save-steps", type=int, default=1000, help="Steps between saving checkpoints")
    training_group.add_argument("--warmup-steps", type=int, default=200, help="Warmup steps for learning rate scheduler")
    training_group.add_argument("--logging-steps", type=int, default=100, help="Steps between logging")
    training_group.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    
    # Inference arguments
    inference_group = parser.add_argument_group('Inference Arguments')
    inference_group.add_argument("--model-path", type=str, help="Path to the fine-tuned model")
    inference_group.add_argument("--device", type=str, choices=['cuda', 'cpu'], help="Device to run the model on")
    inference_group.add_argument("--clear-history", action="store_true", help="Clear conversation history before processing")
    
    args = parser.parse_args()
    
    # Run in appropriate mode
    if args.train:
        # Validate required training arguments
        if not args.data_path:
            parser.error("--data-path is required for training mode")
            
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Train the model
        train_model(args)
        return 0
    else:  # Inference mode
        return cli_interface(args)

if __name__ == "__main__":
    sys.exit(main())