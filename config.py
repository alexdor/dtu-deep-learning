import os

import torch

dirname = os.path.dirname(__file__)
data_path = os.path.join(dirname, "data")
train_data_path = os.path.join(data_path, "chunked/train/train_*")
valid_data_path = os.path.join(data_path, "chunked/valid/valid_*")
test_data_path = os.path.join(data_path, "chunked/test/test_*")
vocab_path = os.path.join(data_path, "vocab")

save_model_path = os.path.join(data_path, "saved_models")


# Hyperparameters
hidden_dim = 128
emb_dim = 64
batch_size = 32
max_enc_steps = 55  # 99% of the articles are within length 55
max_dec_steps = 15  # 99% of the titles are within length 15
beam_size = 4
min_dec_steps = 3
vocab_size = 50000


lr = 0.001
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4

eps = 1e-12
max_iterations = 500000


intra_encoder = True
intra_decoder = True


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"

PAD_TOKEN = "[PAD]"  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = "[UNK]"  # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = "[START]"  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = "[STOP]"  # This has a vocab id, which is used at the end of untruncated target sequences
