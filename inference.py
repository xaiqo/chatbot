import numpy as np
from config import (
    MAX_RESPONSE_LENGTH,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    MODEL_SAVE_PATH,
    TOKENIZER_SAVE_PATH
)
from core.transformer import Transformer
from training.tokenization.bpe import BPETokenizer
from utils.helpers import softmax, top_k_top_p_filtering


class ChatbotInference:
    def __init__(self, model_path=MODEL_SAVE_PATH, tokenizer_path=TOKENIZER_SAVE_PATH):
        """
        Initialize the chatbot inference.

        Args:
            model_path: Path to the saved model
            tokenizer_path: Path to the saved tokenizer
        """
        self.tokenizer = BPETokenizer()
        self.tokenizer.load(tokenizer_path)

        # Initialize model with the same parameters as during training
        self.model = Transformer(
            vocab_size=self.tokenizer.vocab_size,
            embed_size=512,  # These should match your training config
            num_blocks=6,
            heads=8,
            hidden_size=2048,
            dropout_rate=0.1
        )

        # Load trained weights
        self.model.load(model_path)

        # Special tokens
        self.eos_token_id = self.tokenizer.special_tokens.get("<EOS>", 1)
        self.sos_token_id = self.tokenizer.special_tokens.get("< SOS >", 3)

    def generate_response(self, question, max_length=MAX_RESPONSE_LENGTH,
                          temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P):
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
        # Encode the question
        input_ids = self.tokenizer.encode(question)

        # Add SOS token at the beginning if it's not there
        if input_ids[0] != self.sos_token_id:
            input_ids = [self.sos_token_id] + input_ids

        # Generate response token by token
        for _ in range(max_length):
            # Get model prediction (forward pass)
            logits = self.model.forward(np.array([input_ids]), training=False)[0]

            # Get the last token's logits
            next_token_logits = logits[-1]

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Apply top-k and top-p filtering
            filtered_logits = top_k_top_p_filtering(
                next_token_logits.copy(), top_k=top_k, top_p=top_p
            )

            # Sample from the filtered distribution
            probs = softmax(filtered_logits)
            next_token = np.random.choice(len(probs), p=probs)

            # Add the chosen token to the sequence
            input_ids.append(next_token)

            # Stop if we hit the EOS token
            if next_token == self.eos_token_id:
                break

        # Decode the response (exclude the question tokens)
        question_len = len(self.tokenizer.encode(question))
        response_ids = input_ids[question_len:]
        response = self.tokenizer.decode(response_ids)

        return response

    def answer_question(self, question):
        """
        Process a user question and return an answer.

        Args:
            question: User's question

        Returns:
            str: Answer to the question
        """
        # Format the question for the model
        formatted_question = f"Question: {question}\nAnswer:"

        # Generate response
        answer = self.generate_response(formatted_question)

        return answer.strip()