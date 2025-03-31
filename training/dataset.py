import numpy as np
import os
import glob


class TextDataset:
    def __init__(self, data_dir, tokenizer, seq_length=64):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.texts = self._read_text_files(data_dir)
        self.token_ids = self._tokenize_texts()

    def _read_text_files(self, data_dir):
        """Czyta dane tekstowe z plików w katalogu"""
        data_files = glob.glob(os.path.join(data_dir, "*_processed.txt"))
        if not data_files:
            raise FileNotFoundError(f"Nie znaleziono przetworzonych plików w {data_dir}")

        text_data = []
        for file_path in data_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                text_data.extend(f.readlines())

        return text_data

    def _tokenize_texts(self):
        all_tokens = []
        for text in self.texts:
            tokens = self.tokenizer.encode(text)
            all_tokens.extend(tokens)
        return np.array(all_tokens)

    def get_batches(self, batch_size=32):
        """Generuje batche danych treningowych"""
        inputs = []
        targets = []

        if len(self.token_ids) <= self.seq_length + 1:
            raise ValueError("Za mało danych do utworzenia sekwencji treningowych")

        for i in range(0, len(self.token_ids) - self.seq_length - 1, self.seq_length):
            if i + self.seq_length + 1 > len(self.token_ids):
                break

            seq_in = self.token_ids[i:i + self.seq_length]
            seq_out = self.token_ids[i + 1:i + self.seq_length + 1]

            inputs.append(seq_in)
            targets.append(seq_out)

            if len(inputs) >= batch_size:
                yield np.array(inputs), np.array(targets)
                inputs = []
                targets = []

        if len(inputs) > 0:
            yield np.array(inputs), np.array(targets)