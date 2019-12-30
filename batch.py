import logging

import numpy as np
import torch

from config import (
    DEVICE,
    PAD_TOKEN,
    START_DECODING,
    STOP_DECODING,
    batch_size,
    data_path,
    max_dec_steps,
    max_enc_steps,
)
from vocab import abstract2ids, article2ids, example_generator, text_generator


class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(self, data_list, pad_index):
        self.batch_size = batch_size
        self.pad_index = pad_index
        self.data_list = data_list
        self.init_encoder_seq()  # initialize the input to the encoder
        self.init_decoder_seq()  # initialize the input and targets for the decoder
        self.store_orig_strings()
        self.nseqs = self.enc_batch.size(0)

    def init_encoder_seq(self):
        # Determine the maximum length of the encoder input sequence in this batch
        self.max_enc_seq_len = max([data.enc_len for data in self.data_list])
        # Pad the encoder input sequences up to the length of the longest sequence

        for data in self.data_list:
            data.pad_encoder_input(self.max_enc_seq_len, self.pad_index)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
        self.enc_batch = np.zeros(
            (self.batch_size, self.max_enc_seq_len), dtype=np.int32
        )
        self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.zeros(
            (self.batch_size, self.max_enc_seq_len), dtype=np.float32
        )

        # Fill in the numpy arrays
        for i, data in enumerate(self.data_list):
            self.enc_batch[i, :] = data.enc_input[:]
            self.enc_lens[i] = data.enc_len
            for j in range(data.enc_len):
                self.enc_padding_mask[i][j] = 1
        self.enc_padding_mask = (
            torch.from_numpy(self.enc_padding_mask).long().to(DEVICE)
        )
        self.enc_batch = torch.from_numpy(self.enc_batch).long().to(DEVICE)
        # For pointer-generator mode, need to store some extra info
        # Determine the max number of in-article OOVs in this batch
        self.max_art_oovs = max(
            [len(data.article_oovs) for data in self.data_list]
        )
        # Store the in-article OOVs themselves
        self.art_oovs = [data.article_oovs for data in self.data_list]
        # Store the version of the enc_batch that uses the article OOV ids
        self.enc_batch_extend_vocab = np.zeros(
            (self.batch_size, self.max_enc_seq_len), dtype=np.int32
        )
        for i, data in enumerate(self.data_list):
            self.enc_batch_extend_vocab[i, :] = data.enc_input_extend_vocab[:]
        self.enc_batch_extend_vocab = (
            torch.from_numpy(self.enc_batch_extend_vocab).long().to(DEVICE)
        )

    def init_decoder_seq(self):
        # Pad the inputs and targets
        for data in self.data_list:
            data.pad_decoder_inp_targ(max_dec_steps, self.pad_index)

        # Initialize the numpy arrays.
        self.dec_batch = np.zeros(
            (self.batch_size, max_dec_steps), dtype=np.int32
        )
        self.target_batch = np.zeros(
            (self.batch_size, max_dec_steps), dtype=np.int32
        )
        self.dec_lens = np.zeros((self.batch_size), dtype=np.int32)
        # Fill in the numpy arrays
        for i, data in enumerate(self.data_list):
            self.dec_batch[i, :] = data.dec_input[:]
            self.target_batch[i, :] = data.target[:]
            self.dec_lens[i] = data.dec_len
        self.dec_batch = torch.from_numpy(self.dec_batch).long().to(DEVICE)
        self.target_batch = (
            torch.from_numpy(self.target_batch).long().to(DEVICE)
        )
        self.dec_mask = self.dec_batch != self.pad_index
        self.ntokens = (self.target_batch != self.pad_index).sum()

    def store_orig_strings(self):
        self.original_articles = [
            data.article for data in self.data_list
        ]  # list of lists
        self.original_abstracts = [
            data.abstract for data in self.data_list
        ]  # list of lists
        self.original_abstracts_sents = [
            data.abstract_sentences for data in self.data_list
        ]  # list of list of lists


class BatchGeneration:
    def __init__(self, vocab, data_path, mode="train"):
        self.single_pass = mode != "train"
        self.data_path = data_path
        self.vocab = vocab
        self._finished_reading = False
        # Get ids of special tokens
        self.pad_index = vocab.word2id(PAD_TOKEN)
        self.start_decoding = vocab.word2id(START_DECODING)
        self.stop_decoding = vocab.word2id(STOP_DECODING)

    def data_gen(self):
        generator = example_generator(self.data_path, self.single_pass)
        while True:
            data_list = []
            for i in range(batch_size):
                res = text_generator(generator)
                if self._finished_reading:
                    return
                try:
                    article, abstract_sentences = next(self.get_src_trg(res))
                except StopIteration:
                    yield Batch(
                        data_list, self.pad_index,
                    )

                data_list.append(self.encode_data(article, abstract_sentences))
            yield Batch(
                data_list, self.pad_index,
            )

    def encode_data(self, article, abstract_sentences):
        # Process the article
        article_words = article.split()
        if len(article_words) > max_enc_steps:
            article_words = article_words[:max_enc_steps]
        enc_len = len(
            article_words
        )  # store the length after truncation but before padding
        enc_input = [
            self.vocab.word2id(w) for w in article_words
        ]  # list of word ids; OOVs are represented by the id for UNK token

        # Process the abstract
        abstract = " ".join(abstract_sentences)  # string
        abstract_words = abstract.split()  # list of strings
        abs_ids = [
            self.vocab.word2id(w) for w in abstract_words
        ]  # list of word ids; OOVs are represented by the id for UNK token

        # Get the decoder input sequence and target sequence
        dec_input, _ = self.get_dec_inp_targ_seqs(
            abs_ids, max_dec_steps, self.start_decoding, self.stop_decoding,
        )

        # If using pointer-generator mode, we need to store some extra info
        # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
        enc_input_extend_vocab, article_oovs = article2ids(
            article_words, self.vocab
        )

        # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
        abs_ids_extend_vocab = abstract2ids(
            abstract_words, self.vocab, article_oovs
        )

        # Get decoder target sequence
        _, target = self.get_dec_inp_targ_seqs(
            abs_ids_extend_vocab,
            max_dec_steps,
            self.start_decoding,
            self.stop_decoding,
        )

        # Store the original strings
        return Data(
            article=article,
            abstract=abstract,
            abstract_sentences=abstract_sentences,
            target=target,
            enc_input_extend_vocab=enc_input_extend_vocab,
            article_oovs=article_oovs,
            dec_input=dec_input,
            enc_len=enc_len,
            enc_input=enc_input,
        )

    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:  # truncate
            inp = inp[:max_len]
            target = target[:max_len]  # no end_token
        else:  # no truncation
            target.append(stop_id)  # end token
        assert len(inp) == len(target)
        return inp, target

    def get_src_trg(self, res):
        while True:
            try:
                (article, abstract) = next(
                    res
                )  # read the next example from file. article and abstract are both strings.
            except Exception:  # if there are no more examples:
                logging.info(
                    "The example generator for this example queue filling thread has exhausted data."
                )
                if self.single_pass:
                    logging.info(
                        "single_pass mode is on, so we've finished reading dataset. This thread is stopping."
                    )
                    self._finished_reading = True
                    break
                else:
                    raise Exception(
                        "single_pass mode is off but the example generator is out of data; error."
                    )

            # abstract_sentences = [sent.strip() for sent in data.abstract2sents(abstract)] # Use the <s> and </s> tags in abstract to get a list of sentences.
            abstract_sentences = [abstract.strip()]
            yield (article, abstract_sentences)


class Data:
    def __init__(
        self,
        article,
        abstract,
        abstract_sentences,
        target,
        enc_input_extend_vocab,
        article_oovs,
        dec_input,
        enc_len,
        enc_input,
    ):
        self.dec_len = len(dec_input)
        self.article = article
        self.abstract = abstract
        self.abstract_sentences = abstract_sentences
        self.target = target
        self.enc_input_extend_vocab = enc_input_extend_vocab
        self.article_oovs = article_oovs
        self.dec_input = dec_input
        self.enc_len = enc_len
        self.enc_input = enc_input

    def pad_decoder_inp_targ(self, max_len, pad_id):
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

    def pad_encoder_input(self, max_len, pad_id):
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        while len(self.enc_input_extend_vocab) < max_len:
            self.enc_input_extend_vocab.append(pad_id)
