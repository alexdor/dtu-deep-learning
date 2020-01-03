import gzip
import re
from collections import Counter
from functools import lru_cache
from os import path
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from numpy.random.mtrand import shuffle

from vocab import Vocab


def simple_tokenizer(
    text: str, lower: bool = False, newline: str = None
) -> List[str]:
    """Split an already tokenized input `text`."""
    if lower:
        text = text.lower()
    if newline is not None:  # replace newline by a token
        text = text.replace("\n", " " + newline + " ")
    return text.split()


class Dataset(object):
    def __init__(
        self,
        filename: str,
        tokenize: Callable = simple_tokenizer,
        max_src_len: int = None,
        max_tgt_len: int = None,
        truncate_src: bool = False,
        truncate_tgt: bool = False,
    ):
        print("Reading dataset %s..." % filename, end=" ", flush=True)
        self.filename = filename
        self.pairs = []
        self.src_len = 0
        self.tgt_len = 0
        openner = gzip.open if filename.endswith(".gz") else open
        with openner(filename, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                pair = line.strip().split("\t")
                if len(pair) != 2:
                    print("Line %d of %s is malformed." % (i, filename))
                    continue
                src = tokenize(pair[0])
                if max_src_len and len(src) > max_src_len:
                    if truncate_src:
                        src = src[:max_src_len]
                    else:
                        continue
                tgt = tokenize(pair[1])
                if max_tgt_len and len(tgt) > max_tgt_len:
                    if truncate_tgt:
                        tgt = tgt[:max_tgt_len]
                    else:
                        continue
                src_len = len(src) + 1  # EOS
                tgt_len = len(tgt) + 1  # EOS
                self.src_len = max(self.src_len, src_len)
                self.tgt_len = max(self.tgt_len, tgt_len)
                self.pairs.append(Example(src, tgt, src_len, tgt_len))
        print("%d pairs." % len(self.pairs))

    def build_vocab(
        self,
        vocab_size: int = None,
        src: bool = True,
        tgt: bool = True,
        embed_file: str = None,
    ) -> Vocab:
        filename, _ = path.splitext(self.filename)
        if vocab_size:
            filename += ".%d" % vocab_size
        filename += ".vocab"
        if path.isfile(filename):
            vocab = torch.load(filename)
            print("Vocabulary loaded, %d words." % len(vocab))
        else:
            print("Building vocabulary...", end=" ", flush=True)
            vocab = Vocab()
            for example in self.pairs:
                if src:
                    vocab.add_words(example.src)
                if tgt:
                    vocab.add_words(example.tgt)
            vocab.trim(vocab_size=vocab_size)
            print("%d words." % len(vocab))
            torch.save(vocab, filename)
        if embed_file:
            count = vocab.load_embeddings(embed_file)
            print("%d pre-trained embeddings loaded." % count)
        return vocab

    def generator(
        self,
        batch_size: int,
        src_vocab: Vocab = None,
        tgt_vocab: Vocab = None,
        ext_vocab: bool = False,
    ):
        ptr = len(self.pairs)  # make sure to shuffle at first run
        if ext_vocab:
            assert src_vocab is not None
            base_oov_idx = len(src_vocab)
        while True:
            if ptr + batch_size > len(self.pairs):
                shuffle(self.pairs)  # shuffle inplace to save memory
                ptr = 0
            examples = self.pairs[ptr : ptr + batch_size]
            ptr += batch_size
            src_tensor, tgt_tensor = None, None
            lengths, oov_dict = None, None
            if src_vocab or tgt_vocab:
                # initialize tensors
                if src_vocab:
                    examples.sort(key=lambda x: -x.src_len)
                    lengths = [x.src_len for x in examples]
                    max_src_len = lengths[0]
                    src_tensor = torch.zeros(
                        max_src_len, batch_size, dtype=torch.long
                    )
                    if ext_vocab:
                        oov_dict = OOVDict(base_oov_idx)
                if tgt_vocab:
                    max_tgt_len = max(x.tgt_len for x in examples)
                    tgt_tensor = torch.zeros(
                        max_tgt_len, batch_size, dtype=torch.long
                    )
                # fill up tensors by word indices
                for i, example in enumerate(examples):
                    if src_vocab:
                        for j, word in enumerate(example.src):
                            idx = src_vocab[word]
                            if ext_vocab and idx == src_vocab.UNK:
                                idx = oov_dict.add_word(i, word)
                            src_tensor[j, i] = idx
                        src_tensor[example.src_len - 1, i] = src_vocab.EOS
                    if tgt_vocab:
                        for j, word in enumerate(example.tgt):
                            idx = tgt_vocab[word]
                            if ext_vocab and idx == src_vocab.UNK:
                                idx = oov_dict.word2index.get((i, word), idx)
                            tgt_tensor[j, i] = idx
                        tgt_tensor[example.tgt_len - 1, i] = tgt_vocab.EOS
            yield Batch(examples, src_tensor, tgt_tensor, lengths, oov_dict)


class Example(NamedTuple):
    src: List[str]
    tgt: List[str]
    src_len: int  # inclusive of EOS, so that it corresponds to tensor shape
    tgt_len: int  # inclusive of EOS, so that it corresponds to tensor shape


class OOVDict(object):
    def __init__(self, base_oov_idx):
        self.word2index = {}  # type: Dict[Tuple[int, str], int]
        self.index2word = {}  # type: Dict[Tuple[int, int], str]
        self.next_index = {}  # type: Dict[int, int]
        self.base_oov_idx = base_oov_idx
        self.ext_vocab_size = base_oov_idx

    def add_word(self, idx_in_batch, word) -> int:
        key = (idx_in_batch, word)
        index = self.word2index.get(key)
        if index is not None:
            return index
        index = self.next_index.get(idx_in_batch, self.base_oov_idx)
        self.next_index[idx_in_batch] = index + 1
        self.word2index[key] = index
        self.index2word[(idx_in_batch, index)] = word
        self.ext_vocab_size = max(self.ext_vocab_size, index + 1)
        return index


class Batch(NamedTuple):
    examples: List[Example]
    input_tensor: Optional[torch.Tensor]
    target_tensor: Optional[torch.Tensor]
    input_lengths: Optional[List[int]]
    oov_dict: Optional[OOVDict]

    @property
    def ext_vocab_size(self):
        if self.oov_dict is not None:
            return self.oov_dict.ext_vocab_size
        return None
