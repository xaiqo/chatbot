import re
import json
from collections import defaultdict


class BPETokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = defaultdict(int)
        self.token_to_id = {}
        self.id_to_token = {}
        self.cache = {}
        self.special_tokens = {
            "<PAD>": 0,
            "<EOS>": 1,
            "<UNK>": 2,
            "<SOS>": 3,
        }

    def train(self, text, min_frequency=2):
        words = self._preprocess(text)
        self._build_vocab(words, min_frequency)

        for i, token in enumerate(self.special_tokens.keys()):
            self.token_to_id[token] = i
            self.id_to_token[i] = token

        start_idx = len(self.special_tokens)
        for i, token in enumerate(sorted(self.vocab.keys())):
            if token not in self.token_to_id:
                self.token_to_id[token] = i + start_idx
                self.id_to_token[i + start_idx] = token

        self._learn_bpe()

    def _preprocess(self, text):
        """Preprocessuje tekst na listę słów."""
        return re.findall(r'\S+\n?', text)

    def _build_vocab(self, words, min_frequency=2):
        """Buduje słownik częstości dla poszczególnych znaków."""
        word_freqs = defaultdict(int)

        for word in words:
            word_freqs[word] += 1

        for word, freq in word_freqs.items():
            if freq < min_frequency:
                continue

            chars = list(word) + ['</w>']
            for char in chars:
                self.vocab[char] += freq

    def _get_stats(self, vocab=None):
        """Zlicza częstość występowania par znaków."""
        if vocab is None:
            vocab = self.vocab

        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair, vocab):
        """Łączy parę znaków w słowniku."""
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)

        for word, freq in vocab.items():
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = freq

        return new_vocab

    def _learn_bpe(self):
        """Uczenie się BPE poprzez iteracyjne łączenie najpopularniejszych par."""
        vocab = {' '.join(word): freq for word, freq in self.vocab.items()}

        num_merges = self.vocab_size - len(self.token_to_id)
        for i in range(num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            self.merges[best_pair] = i

            vocab = self._merge_vocab(best_pair, vocab)

            new_token = ''.join(best_pair)
            self.token_to_id[new_token] = len(self.token_to_id)
            self.id_to_token[len(self.id_to_token)] = new_token

            if len(self.token_to_id) >= self.vocab_size:
                break

    def encode(self, text):
        """Konwertuje tekst na sekwencję identyfikatorów tokenów."""
        tokens = []
        for word in self._preprocess(text):
            word_tokens = self._tokenize_word(word)
            tokens.extend([self.token_to_id.get(token, self.special_tokens["<UNK>"]) for token in word_tokens])

        tokens.append(self.special_tokens["<EOS>"])
        return tokens

    def _tokenize_word(self, word):
        """Tokenizuje pojedyncze słowo za pomocą wyuczonych operacji BPE."""
        if word in self.cache:
            return self.cache[word]

        chars = list(word) + ['</w>']
        word_tokens = [' '.join(chars)]

        for pair, _ in sorted(self.merges.items(), key=lambda x: x[1]):
            new_tokens = []
            for token in word_tokens:
                parts = token.split()
                i = 0
                while i < len(parts) - 1:
                    if (parts[i], parts[i + 1]) == pair:
                        new_tokens.append(parts[i] + parts[i + 1])
                        i += 2
                    else:
                        new_tokens.append(parts[i])
                        i += 1
                if i < len(parts):
                    new_tokens.append(parts[i])

                word_tokens = new_tokens

        self.cache[word] = word_tokens
        return word_tokens

    def decode(self, token_ids):
        """Konwertuje sekwencję identyfikatorów tokenów z powrotem na tekst."""
        tokens = [self.id_to_token.get(idx, "<UNK>") for idx in token_ids]
        tokens = [t for t in tokens if t not in self.special_tokens.keys()]

        text = ''.join(tokens).replace('</w>', ' ')
        return text.strip()

    def save(self, filepath):
        """Zapisuje tokenizer do pliku JSON."""
        data = {
            "vocab_size": self.vocab_size,
            "merges": {' '.join(k): v for k, v in self.merges.items()},
            "token_to_id": self.token_to_id,
            "special_tokens": self.special_tokens
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath):
        """Ładuje tokenizer z pliku JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.vocab_size = data["vocab_size"]
        self.merges = {tuple(k.split()): v for k, v in data["merges"].items()}
        self.token_to_id = data["token_to_id"]
        self.special_tokens = data["special_tokens"]
        self.id_to_token = {int(i): t for t, i in self.token_to_id.items()}

        self.cache = {}