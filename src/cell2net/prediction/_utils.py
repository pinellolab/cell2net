import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def random_seq(seq_len):
    bases = ["A", "C", "G", "T"]
    rand_seq = "".join([np.random.choice(bases) for i in range(seq_len)])
    return rand_seq
