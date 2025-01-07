===================
Preprocessing: `pp`
===================
.. currentmodule:: cell2net

ATAC-seq matrix
==========================
.. autosummary::
    :toctree: _autosummary

    preprocessing.binarize

ATAC-seq fragment
==========================
.. autosummary::
    :toctree: _autosummary

    preprocessing.calculate_depth
    preprocessing.collapse_consecutive_values
    preprocessing.fragments_to_coverage
    preprocessing.fragment_to_bigwig
    preprocessing.split_fragments

Gene
=============================
.. autosummary::
    :toctree: _autosummary

    preprocessing.get_gene_tss_coord
    preprocessing.add_gene_tss_coord

Motif
=============================
.. autosummary::
    :toctree: _autosummary

    preprocessing.get_motifs_from_jaspar
    preprocessing.match_motif

Peaks
=============================
.. autosummary::
    :toctree: _autosummary

    preprocessing.add_peaks
    preprocessing.peak_to_gene
    preprocessing.add_dna_sequence