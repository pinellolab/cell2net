import os
import random

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def one_hot_encode(seq):
    # Make sure seq has only allowed bases
    allowed = set("ACTGN")
    if not set(seq).issubset(allowed):
        invalid = set(seq) - allowed
        raise ValueError(f"Sequence contains chars not in allowed DNA alphabet (ACGTN): {invalid}")

    # Dictionary returning one-hot encoding for each nucleotide
    nuc_d = {
        "A": [1.0, 0.0, 0.0, 0.0],
        "C": [0.0, 1.0, 0.0, 0.0],
        "G": [0.0, 0.0, 1.0, 0.0],
        "T": [0.0, 0.0, 0.0, 1.0],
        "N": [0.0, 0.0, 0.0, 0.0],
    }

    # Create array from nucleotide sequence
    vec = torch.tensor([nuc_d[x] for x in seq], dtype=torch.float32)

    return vec


def random_seq(seq_len):
    bases = ["A", "C", "G", "T"]
    rand_seq = "".join([np.random.choice(bases) for i in range(seq_len)])
    return rand_seq
