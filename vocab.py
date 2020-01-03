import re
from collections import Counter
from functools import lru_cache
from os import path
from typing import List, Tuple

import numpy as np


class Vocab(object):
    word_detector = re.compile("\w")
    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3

    def __init__(self):
        self.word2index = {}
        self.word2count = Counter()
        self.reserved = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        self.index2word = self.reserved[:]
        self.embeddings = None

    def add_words(self, words: List[str]):
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = len(self.index2word)
                self.index2word.append(word)
        self.word2count.update(words)

    def trim(self, *, vocab_size: int = None, min_freq: int = 1):
        if min_freq <= 1 and (
            vocab_size is None or vocab_size >= len(self.word2index)
        ):
            return
        ordered_words = sorted(
            ((c, w) for (w, c) in self.word2count.items()), reverse=True
        )
        if vocab_size:
            ordered_words = ordered_words[:vocab_size]
        self.word2index = {}
        self.word2count = Counter()
        self.index2word = self.reserved[:]
        for count, word in ordered_words:
            if count < min_freq:
                break
            self.word2index[word] = len(self.index2word)
            self.word2count[word] = count
            self.index2word.append(word)

    def load_embeddings(self, file_path: str, dtype=np.float32) -> int:
        num_embeddings = 0
        vocab_size = len(self)
        with open(file_path, "rb", encoding="utf-8") as f:
            for line in f:
                line = line.split()
                word = line[0].decode("utf-8")
                idx = self.word2index.get(word)
                if idx is not None:
                    vec = np.array(line[1:], dtype=dtype)
                    if self.embeddings is None:
                        n_dims = len(vec)
                        self.embeddings = np.random.normal(
                            np.zeros((vocab_size, n_dims))
                        ).astype(dtype)
                        self.embeddings[self.PAD] = np.zeros(n_dims)
                    self.embeddings[idx] = vec
                    num_embeddings += 1
        return num_embeddings

    def __getitem__(self, item):
        if type(item) is int:
            return self.index2word[item]
        return self.word2index.get(item, self.UNK)

    def __len__(self):
        return len(self.index2word)

    @lru_cache(maxsize=None)
    def is_word(self, token_id: int) -> bool:
        """Return whether the token at `token_id` is a word; False for punctuations."""
        if token_id < 4:
            return False
        if token_id >= len(self):
            return True  # OOV is assumed to be words
        token_str = self.index2word[token_id]
        if not self.word_detector.search(token_str) or token_str == "<P>":
            return False
        return True
