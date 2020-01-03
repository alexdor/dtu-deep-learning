import gzip
import os
import re
import subprocess
from collections import Counter
from functools import lru_cache
from multiprocessing.dummy import Pool
from random import shuffle
from tempfile import TemporaryDirectory
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

plt.switch_backend("agg")


class Hypothesis(object):
    def __init__(
        self,
        tokens,
        log_probs,
        dec_hidden,
        dec_states,
        enc_attn_weights,
        num_non_words,
    ):
        self.tokens = tokens  # type: List[int]
        self.log_probs = log_probs  # type: List[float]
        self.dec_hidden = dec_hidden  # shape: (1, 1, hidden_size)
        self.dec_states = dec_states  # list of dec_hidden
        self.enc_attn_weights = (
            enc_attn_weights  # list of shape: (1, 1, src_len)
        )
        self.num_non_words = num_non_words  # type: int

    def __repr__(self):
        return repr(self.tokens)

    def __len__(self):
        return len(self.tokens) - self.num_non_words

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.log_probs)

    def create_next(
        self, token, log_prob, dec_hidden, add_dec_states, enc_attn, non_word
    ):
        return Hypothesis(
            tokens=self.tokens + [token],
            log_probs=self.log_probs + [log_prob],
            dec_hidden=dec_hidden,
            dec_states=self.dec_states + [dec_hidden]
            if add_dec_states
            else self.dec_states,
            enc_attn_weights=self.enc_attn_weights + [enc_attn]
            if enc_attn is not None
            else self.enc_attn_weights,
            num_non_words=self.num_non_words + 1
            if non_word
            else self.num_non_words,
        )


def show_plot(
    loss, step=1, val_loss=None, val_metric=None, val_step=1, file_prefix=None
):
    plt.figure()
    fig, ax = plt.subplots(figsize=(12, 8))
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    ax.set_ylabel("Loss", color="b")
    ax.set_xlabel("Batch")
    plt.plot(range(step, len(loss) * step + 1, step), loss, "b")
    if val_loss:
        plt.plot(
            range(val_step, len(val_loss) * val_step + 1, val_step),
            val_loss,
            "g",
        )
    if val_metric:
        ax2 = ax.twinx()
        ax2.plot(
            range(val_step, len(val_metric) * val_step + 1, val_step),
            val_metric,
            "r",
        )
        ax2.set_ylabel("ROUGE", color="r")
    if file_prefix:
        plt.savefig(file_prefix + ".png")
        plt.close()


def show_attention_map(src_words, pred_words, attention, pointer_ratio=None):
    fig, ax = plt.subplots(figsize=(16, 4))
    im = plt.pcolormesh(np.flipud(attention), cmap="GnBu")
    # set ticks and labels
    ax.set_xticks(np.arange(len(src_words)) + 0.5)
    ax.set_xticklabels(src_words, fontsize=14)
    ax.set_yticks(np.arange(len(pred_words)) + 0.5)
    ax.set_yticklabels(reversed(pred_words), fontsize=14)
    if pointer_ratio is not None:
        ax1 = ax.twinx()
        ax1.set_yticks(
            np.concatenate([np.arange(0.5, len(pred_words)), [len(pred_words)]])
        )
        ax1.set_yticklabels("%.3f" % v for v in np.flipud(pointer_ratio))
        ax1.set_ylabel("Copy probability", rotation=-90, va="bottom")
    # let the horizontal axes labelling appear on top
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    # rotate the tick labels and set their alignment
    plt.setp(
        ax.get_xticklabels(), rotation=-45, ha="right", rotation_mode="anchor"
    )


non_word_char_in_word = re.compile(r"(?<=\w)\W(?=\w)")
not_for_output = {"<PAD>", "<SOS>", "<EOS>", "<UNK>"}


def format_tokens(
    tokens: List[str], newline: str = "<P>", for_rouge: bool = False
) -> str:
    """Join output `tokens` for ROUGE evaluation."""
    tokens = filter(lambda t: t not in not_for_output, tokens)
    if for_rouge:
        tokens = [
            non_word_char_in_word.sub("", t) for t in tokens
        ]  # "n't" => "nt"
    if newline is None:
        s = " ".join(tokens)
    else:  # replace newline tokens by newlines
        lines, line = [], []
        for tok in tokens:
            if tok == newline:
                if line:
                    lines.append(" ".join(line))
                line = []
            else:
                line.append(tok)
        if line:
            lines.append(" ".join(line))
        s = "\n".join(lines)
    return s


def format_rouge_scores(rouge_result: Dict[str, float]) -> str:
    lines = []
    line, prev_metric = [], None
    for key in sorted(rouge_result.keys()):
        metric = key.rsplit("_", maxsplit=1)[0]
        if metric != prev_metric and prev_metric is not None:
            lines.append("\t".join(line))
            line = []
        line.append("%s %s" % (key, rouge_result[key]))
        prev_metric = metric
    lines.append("\t".join(line))
    return "\n".join(lines)


this_dir = os.path.dirname(os.path.abspath(__file__))

rouge_pattern = re.compile(
    rb"(\d+) ROUGE-(.+) Average_([RPF]): ([\d.]+) "
    rb"\(95%-conf\.int\. ([\d.]+) - ([\d.]+)\)"
)


def rouge(
    target: List[List[str]], *predictions: List[List[str]]
) -> List[Dict[str, float]]:
    """Perform single-reference ROUGE evaluation of one or more systems' predictions."""
    results = [
        dict() for _ in range(len(predictions))
    ]  # e.g. 0 => 'su4_f' => 0.35
    with TemporaryDirectory() as folder:  # on my server, /tmp is a RAM disk
        # write SPL files
        eval_entries = []
        for i, tgt_tokens in enumerate(target):
            sys_entries = []
            for j, pred_docs in enumerate(predictions):
                sys_file = "sys%d_%d.spl" % (j, i)
                sys_entries.append('\n    <P ID="%d">%s</P>' % (j, sys_file))
                with open(os.path.join(folder, sys_file), "wt") as f:
                    f.write(format_tokens(pred_docs[i], for_rouge=True))
            ref_file = "ref_%d.spl" % i
            with open(os.path.join(folder, ref_file), "wt") as f:
                f.write(format_tokens(tgt_tokens, for_rouge=True))
            eval_entry = """
<EVAL ID="{1}">
  <PEER-ROOT>{0}</PEER-ROOT>
  <MODEL-ROOT>{0}</MODEL-ROOT>
  <INPUT-FORMAT TYPE="SPL"></INPUT-FORMAT>
  <PEERS>{2}
  </PEERS>
  <MODELS>
    <M ID="A">{3}</M>
  </MODELS>
</EVAL>""".format(
                folder, i, "".join(sys_entries), ref_file
            )
            eval_entries.append(eval_entry)
        # write config file
        xml = '<ROUGE-EVAL version="1.0">{0}\n</ROUGE-EVAL>'.format(
            "".join(eval_entries)
        )
        config_path = os.path.join(folder, "task.xml")
        with open(config_path, "wt") as f:
            f.write(xml)
        # run ROUGE
        out = subprocess.check_output(
            "./ROUGE-1.5.5.pl -e data -a -n 2 -2 4 -u " + config_path,
            shell=True,
            cwd=os.path.join(this_dir, "data"),
        )
    # parse ROUGE output
    for line in out.split(b"\n"):
        match = rouge_pattern.match(line)
        if match:
            sys_id, metric, rpf, value, low, high = match.groups()
            results[int(sys_id)][
                (metric + b"_" + rpf).decode("utf-8").lower()
            ] = float(value)
    return results
