===================
Preprocessing: `pp`
===================
.. currentmodule:: cell2net

ATAC-seq matrix processing
==========================
.. autosummary::
    :toctree: _autosummary

    preprocessing.binarize

ATAC-seq fragments processing
=============================
.. autosummary::
    :toctree: _autosummary

    preprocessing.calculate_depth
    preprocessing.collapse_consecutive_values
    preprocessing.fragments_to_coverage
    preprocessing.fragment_to_bigwig
    preprocessing.split_fragments

Gene processing
=============================
.. autosummary::
    :toctree: _autosummary

    preprocessing.add_gene_tss_coord
    preprocessing.add_gene_tss_coord

Motif processing
=============================
.. autosummary::
    :toctree: _autosummary

    preprocessing.get_motifs_from_jaspar
    preprocessing.match_motif