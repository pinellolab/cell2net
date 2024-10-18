import os


def fragments_to_bigwig(
    fragment_file: str, groupby: list[str], out_dir: str, genome_sizes: dict
):

    assert os.path.exists(fragment_file), print(f"Cannot find file {fragment_file}")

    return NotImplemented
