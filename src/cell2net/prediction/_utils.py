import torch

from cell2net.preprocessing import seq_to_one_hot


def encode_seq(seq_list):
    data = []
    for seq in seq_list:
        data.append(seq_to_one_hot(seq))

    data = torch.stack(data)

    return data
