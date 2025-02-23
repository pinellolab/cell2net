import matplotlib.pyplot as plt

from cell2net._logging import logger


def motif_logo():

    # check if logomaker is installed
    try:
        import logomaker
    except ImportError:
        logger.error(
            "logomaker is not installed. Please install it with: pip install logomaker"
        )

        return None

    fig, ax = plt.subplots(1, 1, figsize=(4, 2))

    # Create a logo object
    # icm = logomaker.alignment_to_matrix(
    #     alignment=["ACGT", "ACGT", "ACGT", "ACGT", "ACGT", "ACGT", "ACGT", "ACGT"]
    # )
    # logo = logomaker.Logo(icm, ax=ax, show_spines=False, baseline_width=0)
