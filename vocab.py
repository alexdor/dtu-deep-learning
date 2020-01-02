# Most of this file is copied form https://github.com/abisee/pointer-generator/blob/master/data.py

import csv
import glob
import logging
import random
import struct

from config import (
    PAD_TOKEN,
    SENTENCE_END,
    SENTENCE_START,
    START_DECODING,
    STOP_DECODING,
    UNKNOWN_TOKEN,
)


# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.
class Vocab(object):
    def __init__(self, vocab_file, max_size):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # keeps track of total number of words in the Vocab

        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
        for w in [PAD_TOKEN, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file, "r", encoding="utf8") as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    # print ('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
                    continue
                w = pieces[0]
                if w in [
                    SENTENCE_START,
                    SENTENCE_END,
                    UNKNOWN_TOKEN,
                    PAD_TOKEN,
                    START_DECODING,
                    STOP_DECODING,
                ]:
                    raise Exception(
                        "<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn't be in the vocab file, but %s is"
                        % w
                    )
                if w in self._word_to_id:
                    raise Exception(
                        "Duplicated word in vocabulary file: %s" % w
                    )
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    # print ("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count))
                    break

        # print ("Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count-1]))

    def word2id(self, word):
        return self._word_to_id.get(word, self._word_to_id[UNKNOWN_TOKEN])

    def id2word(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError("Id not found in vocab: %d" % word_id)
        return self._id_to_word[word_id]

    def size(self):
        return self._count

    def write_metadata(self, fpath):
        print("Writing word embedding metadata file to %s..." % (fpath))
        with open(fpath, "w") as f:
            fieldnames = ["word"]
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            for i in range(self.size()):
                writer.writerow({"word": self._id_to_word[i]})


def article2ids(article_words, vocab):
    ids = []
    oovs = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in article_words:
        i = vocab.word2id(w)
        if i == unk_id:  # If w is OOV
            if w not in oovs:  # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(
                w
            )  # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(
                vocab.size() + oov_num
            )  # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oovs


def abstract2ids(abstract_words, vocab, article_oovs):
    ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id:  # If w is an OOV word
            if w in article_oovs:  # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(
                    w
                )  # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else:  # If w is an out-of-article OOV
                ids.append(unk_id)  # Map to the UNK token id
        else:
            ids.append(i)
    return ids


def get_ukn_word(i, vocab, article_oovs):
    assert (
        article_oovs is not None
    ), "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
    article_oov_idx = i - vocab.size()
    w = UNKNOWN_TOKEN
    try:
        w = article_oovs[article_oov_idx]
    except ValueError as e:  # i doesn't correspond to an article oov
        raise ValueError(
            "Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs"
            % (i, article_oov_idx, len(article_oovs))
        )
    return w


def outputids2words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i)  # might be [UNK]
        except ValueError as e:  # w is OOV
            w = get_ukn_word(i, vocab, article_oovs)
        words.append(w)
    return words


def abstract2sents(abstract):
    cur = 0
    sents = []
    while True:
        try:
            start_p = abstract.index(SENTENCE_START, cur)
            end_p = abstract.index(SENTENCE_END, start_p + 1)
            cur = end_p + len(SENTENCE_END)
            sents.append(abstract[start_p + len(SENTENCE_START) : end_p])
        except ValueError as e:  # no more sentences
            return sents
