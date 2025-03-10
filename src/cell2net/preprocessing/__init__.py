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
    filter_motifs_by_genes,
    get_motifs_from_jaspar,
    match_motif,
    match_motif_with_variants,
    tf_to_gene,
)
from ._peak import add_peaks, peak_to_gene
from ._sequence import add_dna_sequence
from ._utils import bgzip, tabix_index
from ._variants import add_genomic_variants
