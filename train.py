import math
import os
from test import eval_batch, eval_batch_output

import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from dataset import Batch, Dataset
from model import DEVICE, Seq2Seq
from params import Params
from utils import show_plot
from vocab import Vocab

filename_format = "%s.%02d.pt"


class Trainer:
    def __init__(
        self,
        model: Seq2Seq,
        vocab: Vocab,
        data_generator,
        params: Params,
        saved_state: dict = None,
    ):
        self.model = model
        self.vocab = vocab
        self.criterion = nn.NLLLoss(ignore_index=vocab.PAD)
        self.data_generator = data_generator
        self.params = params
        self.saved_state = saved_state
        self.optimizer = self.get_optimizer()
        self.partial_forcing = self.get_value_with_default(
            params.partial_forcing, True
        )

        self.grad_norm = self.get_value_with_default(params.grad_norm, 0)

        self.show_cover_loss = self.get_value_with_default(
            params.show_cover_loss, False
        )

        self.pack_seq = self.get_value_with_default(params.pack_seq, True)

    def get_value_with_default(self, value, default):
        return default if value is None else value

    def get_optimizer(self):
        if self.saved_state is not None:
            return self.saved_state["optimizer"]

        return (
            optim.Adagrad(
                self.model.parameters(),
                lr=self.params.lr,
                initial_accumulator_value=self.params.adagrad_accumulator,
            )
            if self.params.optimizer == "adagrad"
            else optim.Adam(self.model.parameters(), lr=self.params.lr)
        )

    def train(self, valid_generator=None):
        # variables for plotting
        plot_points_per_epoch = max(math.log(self.params.n_batches, 1.6), 1.0)
        plot_every = round(self.params.n_batches / plot_points_per_epoch)
        plot_losses, cached_losses = [], []
        plot_val_losses, plot_val_metrics = [], []

        total_parameters = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        print("Training %d trainable parameters..." % total_parameters)
        past_epochs = 0
        total_batch_count = 0
        self.model.to(DEVICE)

        if self.saved_state is not None:
            past_epochs = self.saved_state["epoch"]
            total_batch_count = self.saved_state["total_batch_count"]

        if self.params.lr_decay:
            lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                self.params.lr_decay_step,
                self.params.lr_decay,
                past_epochs - 1,
            )
        best_avg_loss, best_epoch_id = float("inf"), None

        for epoch_count in range(1 + past_epochs, self.params.n_epochs + 1):
            if self.params.lr_decay:
                lr_scheduler.step()
            rl_ratio = (
                self.params.rl_ratio
                if epoch_count >= self.params.rl_start_epoch
                else 0
            )
            epoch_loss, epoch_metric = 0, 0
            epoch_avg_loss, valid_avg_loss, valid_avg_metric = None, None, None
            prog_bar = tqdm(
                range(1, self.params.n_batches + 1),
                desc="Epoch %d" % epoch_count,
            )
            self.model.train()

            for batch_count in prog_bar:  # training batches
                if self.params.forcing_decay_type:
                    if self.params.forcing_decay_type == "linear":
                        forcing_ratio = max(
                            0,
                            self.params.forcing_ratio
                            - self.params.forcing_decay * total_batch_count,
                        )
                    elif self.params.forcing_decay_type == "exp":
                        forcing_ratio = self.params.forcing_ratio * (
                            self.params.forcing_decay ** total_batch_count
                        )
                    elif self.params.forcing_decay_type == "sigmoid":
                        forcing_ratio = (
                            self.params.forcing_ratio
                            * self.params.forcing_decay
                            / (
                                self.params.forcing_decay
                                + math.exp(
                                    total_batch_count
                                    / self.params.forcing_decay
                                )
                            )
                        )
                    else:
                        raise ValueError(
                            "Unrecognized forcing_decay_type: "
                            + self.params.forcing_decay_type
                        )
                else:
                    forcing_ratio = self.params.forcing_ratio

                batch = next(self.data_generator)
                loss, metric = self.train_batch(
                    batch,
                    forcing_ratio=forcing_ratio,
                    sample=self.params.sample,
                    rl_ratio=rl_ratio,
                )

                epoch_loss += float(loss)
                epoch_avg_loss = epoch_loss / batch_count
                if (
                    metric is not None
                ):  # print ROUGE as well if reinforcement learning is enabled
                    epoch_metric += metric
                    epoch_avg_metric = epoch_metric / batch_count
                    prog_bar.set_postfix(
                        loss="%g" % epoch_avg_loss,
                        rouge="%.4g" % (epoch_avg_metric * 100),
                    )
                else:
                    prog_bar.set_postfix(loss="%g" % epoch_avg_loss)

                cached_losses.append(loss)
                total_batch_count += 1
                if total_batch_count % plot_every == 0:
                    period_avg_loss = sum(cached_losses) / len(cached_losses)
                    plot_losses.append(period_avg_loss)
                    cached_losses = []

            if valid_generator is not None:  # validation batches
                valid_loss, valid_metric = 0, 0
                prog_bar = tqdm(
                    range(1, self.params.n_val_batches + 1),
                    desc="Valid %d" % epoch_count,
                )
                self.model.eval()

                for batch_count in prog_bar:
                    batch = next(valid_generator)
                    loss, metric = eval_batch(
                        batch,
                        self.model,
                        self.vocab,
                        self.criterion,
                        pack_seq=self.pack_seq,
                        show_cover_loss=self.show_cover_loss,
                    )
                    valid_loss += loss
                    valid_metric += metric
                    valid_avg_loss = valid_loss / batch_count
                    valid_avg_metric = valid_metric / batch_count
                    prog_bar.set_postfix(
                        loss="%g" % valid_avg_loss,
                        rouge="%.4g" % (valid_avg_metric * 100),
                    )

                plot_val_losses.append(valid_avg_loss)
                plot_val_metrics.append(valid_avg_metric)

                metric_loss = (
                    -valid_avg_metric
                )  # choose the best model by ROUGE instead of loss
                if metric_loss < best_avg_loss:
                    best_epoch_id = epoch_count
                    best_avg_loss = metric_loss

            else:  # no validation, "best" is defined by training loss
                if epoch_avg_loss < best_avg_loss:
                    best_epoch_id = epoch_count
                    best_avg_loss = epoch_avg_loss

            if self.params.model_path_prefix:
                # save model
                filename = filename_format % (
                    self.params.model_path_prefix,
                    epoch_count,
                )
                torch.save(self.model, filename)
                if (
                    not self.params.keep_every_epoch
                ):  # clear previously saved models
                    for epoch_id in range(1 + past_epochs, epoch_count):
                        if epoch_id != best_epoch_id:
                            try:
                                prev_filename = filename_format % (
                                    self.params.model_path_prefix,
                                    epoch_id,
                                )
                                os.remove(prev_filename)
                            except FileNotFoundError:
                                pass
                # save training status
                torch.save(
                    {
                        "epoch": epoch_count,
                        "total_batch_count": total_batch_count,
                        "train_avg_loss": epoch_avg_loss,
                        "valid_avg_loss": valid_avg_loss,
                        "valid_avg_metric": valid_avg_metric,
                        "best_epoch_so_far": best_epoch_id,
                        "params": self.params,
                        "optimizer": self.optimizer,
                    },
                    "%s.train.pt" % self.params.model_path_prefix,
                )

            if rl_ratio > 0:
                self.params.rl_ratio **= self.params.rl_ratio_power

            show_plot(
                plot_losses,
                plot_every,
                plot_val_losses,
                plot_val_metrics,
                self.params.n_batches,
                self.params.model_path_prefix,
            )

    def train_batch(
        self, batch, *, forcing_ratio=0.5, sample=False, rl_ratio: float = 0,
    ):

        input_lengths = None if not self.pack_seq else batch.input_lengths

        self.optimizer.zero_grad()
        input_tensor = batch.input_tensor.to(DEVICE)
        target_tensor = batch.target_tensor.to(DEVICE)
        ext_vocab_size = batch.ext_vocab_size

        out = self.model(
            input_tensor,
            target_tensor,
            input_lengths,
            self.criterion,
            forcing_ratio=forcing_ratio,
            partial_forcing=self.partial_forcing,
            sample=sample,
            ext_vocab_size=ext_vocab_size,
            include_cover_loss=self.show_cover_loss,
        )

        if rl_ratio > 0:
            assert self.vocab is not None
            sample_out = self.model(
                input_tensor,
                saved_out=out,
                criterion=self.criterion,
                sample=True,
                ext_vocab_size=ext_vocab_size,
            )
            baseline_out = self.model(
                input_tensor,
                saved_out=out,
                visualize=False,
                ext_vocab_size=ext_vocab_size,
            )
            scores = eval_batch_output(
                [ex.tgt for ex in batch.examples],
                self.vocab,
                batch.oov_dict,
                sample_out.decoded_tokens,
                baseline_out.decoded_tokens,
            )
            greedy_rouge = scores[1]["l_f"]
            neg_reward = greedy_rouge - scores[0]["l_f"]
            # if sample > baseline, the reward is positive (i.e. good exploration), rl_loss is negative
            rl_loss = neg_reward * sample_out.loss
            rl_loss_value = neg_reward * sample_out.loss_value
            loss = (1 - rl_ratio) * out.loss + rl_ratio * rl_loss
            loss_value = (
                1 - rl_ratio
            ) * out.loss_value + rl_ratio * rl_loss_value
        else:
            loss = out.loss
            loss_value = out.loss_value
            greedy_rouge = None

        loss.backward()
        if self.grad_norm > 0:
            clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()

        target_length = target_tensor.size(0)
        return loss_value / target_length, greedy_rouge


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Train the seq2seq abstractive summarizer."
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        metavar="R",
        help="path to a saved training status (*.train.pt)",
    )
    args, unknown_args = parser.parse_known_args()

    params = Params()
    model = None
    train_status = None

    if params.model_path_prefix is not None:
        Path(params.model_path_prefix).mkdir(parents=True, exist_ok=True)

    if args.resume_from:
        print("Resuming from %s..." % args.resume_from)
        model = torch.load(args.resume_from, map_location=DEVICE)

    if unknown_args:  # allow command line args to override params.py
        params.update(unknown_args)

    dataset = Dataset(
        os.path.abspath(params.data_path),
        max_src_len=params.max_src_len,
        max_tgt_len=params.max_tgt_len,
        truncate_src=params.truncate_src,
        truncate_tgt=params.truncate_tgt,
    )

    if model is None:
        vocab = dataset.build_vocab(
            params.vocab_size, embed_file=params.embed_file
        )
        model = Seq2Seq(vocab, params)
    else:
        vocab = dataset.build_vocab(params.vocab_size)

    train_gen = dataset.generator(
        params.batch_size, vocab, vocab, bool(params.pointer)
    )

    val_gen = None

    if params.val_data_path:
        val_dataset = Dataset(
            os.path.abspath(params.val_data_path),
            max_src_len=params.max_src_len,
            max_tgt_len=params.max_tgt_len,
            truncate_src=params.truncate_src,
            truncate_tgt=params.truncate_tgt,
        )
        val_gen = val_dataset.generator(
            params.val_batch_size, vocab, vocab, bool(params.pointer)
        )

    trainer = Trainer(model, vocab, train_gen, params, train_status)
    trainer.train(val_gen)


if __name__ == "__main__":
    main()
