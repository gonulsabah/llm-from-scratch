import json
import torch


class Tokenizer:
    def __init__(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
            self.unk_token = self.vocab.get('UNK', None)
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        tokens = []
        for word in text.split():
            i = 0
            # example: states
            # state -> 4
            # -s -> 58
            while i < len(word):
                found = False
                for j in range(len(word), i, -1):
                    subword = word[i:j]
                    if subword in self.vocab:
                        tokens.append(self.vocab[subword])
                        i = j
                        found = True
                        break
                if not found:
                    if self.unk_token is not None:
                        tokens.append(self.unk_token)
                    i += 1
            # Add space token at the end of each word
            tokens.append(self.vocab.get(' ', None))
        if not text.endswith(" "):
            tokens.pop()  # Remove the last space token
        return torch.tensor(tokens)

    def decode(self, tokens):
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        return ''.join([reverse_vocab[token] if token in reverse_vocab else 'UNK' for token in tokens])

    def tokenizer(self, text):
        token_ids = self.encode(text)
        token_ids = token_ids.detach().numpy().tolist()
        return [self.reverse_vocab[id] for id in token_ids]
