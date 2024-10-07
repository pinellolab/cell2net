import torch


def encode_seq(seq_list):
    data = []
    for seq in seq_list:
        data.append(one_hot_encode(seq))

    data = torch.stack(data)

    return data


def one_hot_encode(seq):
    # Make sure seq has only allowed bases
    allowed = set("ACTGN")
    if not set(seq).issubset(allowed):
        invalid = set(seq) - allowed
        raise ValueError(
            f"Sequence contains chars not in allowed DNA alphabet (ACGTN): {invalid}"
        )

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
