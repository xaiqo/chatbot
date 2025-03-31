import re

class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}

    def fit(self, text):
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        unique_tokens = list(set(tokens))
        self.vocab = {token: idx for idx, token in enumerate(unique_tokens)}
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}

    def encode(self, text):
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        return [self.vocab.get(token, 0) for token in tokens]

    def decode(self, token_ids):
        return " ".join([self.reverse_vocab.get(idx, "<UNK>") for idx in token_ids])