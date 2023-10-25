import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pandas as pd
from contrastive_rna_representation.gene_dataset import construct_combination_transcript_dict


def main():
    fasta_file_path=(
        "/ssd005/home/phil/Documents/01_projects/"
        "contrastive_rna_representation/"
        "annotation_data/ref_genomes"
    )
    refseq_file_path=(
        "/ssd005/home/phil/Documents/01_projects/"
        "contrastive_rna_representation/annotation_data/"
        "refseq_gencode_files"
    )

    transcript_dict = construct_combination_transcript_dict(
        refseq_files=(
            f"{refseq_file_path}/all_celegans_ncbi_refseq_ce11.tsv",
            f"{refseq_file_path}/all_chimp_ncbi_refseq_panTr06.tsv",
            f"{refseq_file_path}/all_cow_ncbi_refseq_bosTau9.tsv",
            f"{refseq_file_path}/all_dog_ncbi_refseq_canFam4.tsv",
            f"{refseq_file_path}/all_drosophila_ncbi_refseq_dm6.tsv",
            f"{refseq_file_path}/all_rat_ncbi_refseq_rn7.tsv",
            f"{refseq_file_path}/all_rhesus_ncbi_refseq_rheMac10.tsv",
            f"{refseq_file_path}/all_spombe_ncbi_refseq_sacCer3.tsv",
            f"{refseq_file_path}/all_zebrafish_ncbi_refseq_danRer11.tsv",
            f"{refseq_file_path}/human_comprehensive_gencode_v41_hg38.tsv",
            f"{refseq_file_path}/mouse_comprehensive_gencodevm25_mm10.tsv",
        ),
        fasta_files=(
            f"{fasta_file_path}/ce11.fa",
            f"{fasta_file_path}/panTro6.fa",
            f"{fasta_file_path}/bosTau9.fa",
            f"{fasta_file_path}/canFam4.fa",
            f"{fasta_file_path}/dm6.fa",
            f"{fasta_file_path}/rn7.fa",
            f"{fasta_file_path}/rheMac10.fa",
            f"{fasta_file_path}/sacCer3.fa",
            f"{fasta_file_path}/danRer11.fa",
            f"{fasta_file_path}/hg38.fa",
            f"{fasta_file_path}/mm10.fa",
        ),
        species_names=(
            'celegans',
            'chimp',
            'cow',
            'dog',
            'drosophila',
            'rat',
            'rhesus',
            'spombe',
            'zebrafish',
            'human',
            'mouse',
        ),
        transcript_length_drop=12288,
        mini_dataset=False,
        do_homolog_map=True,
        drop_non_nm=False,
    )

    eval_normalizing_factor(transcript_dict, constant_term=0.2)
    eval_normalizing_factor(transcript_dict, constant_term=0.4)
    eval_normalizing_factor(transcript_dict, constant_term=0.6)
    eval_normalizing_factor(transcript_dict, constant_term=0.8)
    eval_normalizing_factor(transcript_dict, constant_term=1.0)


def eval_normalizing_factor(transcript_dict, constant_term=0.2):
    n_transcripts_per_gene = np.zeros(len(transcript_dict.keys()), dtype=np.int32)

    for gene_index, (gene, transcripts) in enumerate(
        tqdm(transcript_dict.items(), leave=False)
    ):

        n_transcripts_per_gene[gene_index] = len(transcripts)


    d_single = {key: value for key, value in transcript_dict.items() if len(value) > 1}
    nt_per_g_drop = np.zeros(len(d_single.keys()), dtype=np.int32)

    for gene_index, (gene, transcripts) in enumerate(
        tqdm(d_single.items(), leave=False)
    ):

        nt_per_g_drop[gene_index] = len(transcripts)


    log_nt_per_g = np.log(n_transcripts_per_gene + constant_term)
    normalize_ratio = (log_nt_per_g.sum() / log_nt_per_g.shape[0])
    norm_log_nt_per_g = log_nt_per_g / normalize_ratio
    print(
        "All transcripts",
        constant_term,
        norm_log_nt_per_g,
        normalize_ratio,
        norm_log_nt_per_g[0]/norm_log_nt_per_g[-1]
    )


    log_nt_per_g = np.log(nt_per_g_drop + constant_term)
    normalize_ratio = (log_nt_per_g.sum() / log_nt_per_g.shape[0])
    norm_log_nt_per_g = log_nt_per_g / normalize_ratio
    print(
        "Drop Single",
        constant_term,
        norm_log_nt_per_g,
        normalize_ratio,
        norm_log_nt_per_g[0]/norm_log_nt_per_g[-1]
    )


if __name__ == '__main__':
    main()
