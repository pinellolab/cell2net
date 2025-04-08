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
    preprocessing.filter_motifs_by_genes
    preprocessing.match_motif
    preprocessing.tf_to_gene

Peaks
=============================
.. autosummary::
    :toctree: _autosummary

    preprocessing.add_peaks
    preprocessing.peak_to_gene

Sequences
=============================
.. autosummary::
    :toctree: _autosummary

    preprocessing.add_dna_sequence,
    preprocessing.add_variants_to_sequence,
    preprocessing.dinucleotide_shuffle_one_hot,
    preprocessing.dinucleotide_shuffle_str,
    preprocessing.one_hot_to_seq,
    preprocessing.random_seq,
    preprocessing.seq_to_one_hot,
    preprocessing.update_sequence_with_variants,
