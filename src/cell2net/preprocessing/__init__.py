from ._atac import binarize
from ._fragments import (
    calculate_depth,
    collapse_consecutive_values,
    fragment_to_bigwig,
    fragments_to_coverage,
    split_fragments,
)
from ._gene import add_gene_tss_coord, get_gene_tss_coord
from ._motif import (
    prepare_scaner,
    filter_motifs_by_genes,
    get_tf_motifs,
    match_motif,
    match_motif_with_variants,
    tf_to_gene,
    tf_to_gene_with_variants,
)
from ._peak import add_peaks, peak_to_gene
from ._sequence import (
    add_dna_sequence,
    add_variants_to_sequence,
    dinucleotide_shuffle_one_hot,
    dinucleotide_shuffle_str,
    random_seq,
    one_hot_to_seq,
    seq_to_one_hot,
    update_sequence_with_variants,
)
from ._utils import bgzip, tabix_index
from ._variants import add_genomic_variants, get_genomic_variants
