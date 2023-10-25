__all__ = ["construct_gene_np", "construct_gene_dataset"]

from .data import RefseqDataset, Transcript

import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from typing import Dict
import os
from natsort import natsorted
import glob

import seaborn as sns
import pandas as pd
from .dataset import prepare_refseq_dataset, generate_gene_to_transcript_dict

sns.set()


def construct_gene_np(
    transcript_dict: Dict[str, Transcript],
    pad_length_to: int = 12288,
    zero_mean=True,
    zero_pad=False,
    n_tracks=4,
):
    """_summary_
    Construct a one hot numpy dataset of shape (n_transcripts, pad_length_to, 4)

    Args:
        transcript_dict (_type_): _description_

    Returns:
        t_dataset: (n_transcripts, pad_length_to, 4)
        gene_id: (n_transcripts,)
        n_transcripts_per_gene: (n_genes,)
    """
    # count number of transcripts
    number_of_transcripts = sum([len(x) for x in transcript_dict.values()])
    # construct transcript and gene_id matrix
    t_dataset = np.zeros(
        (number_of_transcripts, pad_length_to, n_tracks), dtype=np.float32
    )
    gene_id = np.zeros((number_of_transcripts))
    n_transcripts_per_gene = np.zeros(len(transcript_dict.keys()), dtype=np.int32)

    global_transcript_index = 0

    for gene_index, (gene, transcripts) in enumerate(
        tqdm(transcript_dict.items(), leave=False)
    ):
        n_transcripts_per_gene[gene_index] = len(transcripts)
        assert n_transcripts_per_gene[gene_index] > 0

        for transcript in transcripts:
            # Encode sequence
            t_dataset[
                global_transcript_index, :, 0:4
            ] = transcript.one_hot_encode_transcript(
                pad_length_to=pad_length_to, zero_mean=zero_mean, zero_pad=zero_pad
            )
            if n_tracks >= 5:
                t_dataset[
                    global_transcript_index, :, 4:5
                ] = transcript.encode_coding_sequence_track(pad_length_to=pad_length_to)
            if n_tracks >= 6:
                t_dataset[
                    global_transcript_index, :, 5:6
                ] = transcript.encode_splice_track(pad_length_to=pad_length_to)

            gene_id[global_transcript_index] = gene_index

            global_transcript_index += 1

    return t_dataset, gene_id, n_transcripts_per_gene


def construct_gene_dataset(
    transcript_length_drop=12288,
    refseq_location="../data/ncbi_refseq_curated_20221003.tsv",
    fasta_file_location="../data/hg38.fa",
    mini_dataset=False,
    zero_mean=True,
    zero_pad=False,
    drop_non_nm=False,
):
    refseq_data = prepare_refseq_dataset(
        transcript_length_drop=transcript_length_drop,
        refseq_location=refseq_location,
        fasta_file_location=fasta_file_location,
        mini_dataset=mini_dataset,
        drop_non_nm=drop_non_nm,
    )
    transcript_dict = generate_gene_to_transcript_dict(refseq_data)

    # Currently has identical pairs
    t_dataset, gene_id, n_transcripts_per_gene = construct_gene_np(
        transcript_dict,
        zero_mean=zero_mean,
        zero_pad=zero_pad,
    )
    ragged_tensors = tf.RaggedTensor.from_value_rowids(
        values=t_dataset, value_rowids=gene_id
    )

    tf_data = tf.data.Dataset.from_tensor_slices(
        (ragged_tensors, n_transcripts_per_gene)
    )

    return tf_data


def batch_transcript_dict(
    transcript_dict, n_genes_per_sub_dict=150, drop_single_t_genes=False
):
    transcript_dict_part = dict()
    transcript_dict.keys()
    transcript_dicts = list()

    for i, (gene, transcripts) in enumerate(transcript_dict.items()):
        # 1500 genes per transcript dict part
        if i % n_genes_per_sub_dict == 0 and i != 0:
            transcript_dicts.append(transcript_dict_part)
            transcript_dict_part = dict()

        # if drop single transcript don't write it
        if drop_single_t_genes and len(transcripts) == 1:
            continue

        # add gene and transcripts to part of transcript_dict
        transcript_dict_part[gene] = transcripts

    return transcript_dicts


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))


def _ints_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value.flatten()))


def rename_gene_names_using_homology_map(
    refseq_data,
    homolog_matching_df="/ssd005/home/phil/Documents/01_projects/contrastive_rna_representation/data/HOM_MouseHumanSequence.rpt",
):
    df = pd.read_csv(homolog_matching_df, sep="\t")

    # Create a new transcript dict with new gene names
    name_to_id = {row["Symbol"]: row["DB Class Key"] for index, row in df.iterrows()}
    new_transcripts = list()
    for transcript in refseq_data.transcripts:
        # if transcript gene is in name to id dict then change the gene name to new name
        if transcript.gene in name_to_id.keys():
            transcript.gene = name_to_id[transcript.gene]

        new_transcripts.append(transcript)
    return RefseqDataset(new_transcripts)

def multi_species_homologene_map(
    refseq_data,
    species_name,
    homologene_path=(
        '/ssd005/home/phil/Documents/01_projects/'
        'contrastive_rna_representation/'
        'annotation_data/homology_maps_homologene'
    )
):
    df = pd.read_csv(f"{homologene_path}/{species_name}_homology_map.csv")
    name_to_id = {
        row["gene_name"]: species_name + "_" + f"{row['gene_group']}" for _, row in df.iterrows()
    }
    new_transcripts = list()
    for transcript in refseq_data.transcripts:
        # if transcript gene is in name to id dict then change the gene name to new name
        if transcript.gene in name_to_id.keys():
            transcript.gene = name_to_id[transcript.gene]

        new_transcripts.append(transcript)
    return RefseqDataset(new_transcripts)


def multi_species_homology_map(
    refseq_data,
    species_name,
    homolog_matching_df="/ssd005/home/phil/Documents/"
    "01_projects/contrastive_rna_representation/annotation_data"
    "/homologene/homologene.data",
):

    df = pd.read_csv(
        homolog_matching_df,
        sep="\t",
        names=['hid', 'tax_id', 'gene_id', 'gene_symbol','protein_id', 'protein_acc'],
    )
    species_to_homologene_id = {
        'mouse': 10090,
        'rat': 10116,
        # 28985,
        # 318829,
        # 33169,
        # 3702,
        # 4530,
        'spombe': 4896,
        # 4932,
        # 5141,
        'celegans': 6239,
        # 7165,
        'drosophila': 7227,
        'zebrafish': 7955,
        # 8364,
        # 9031,
        'rhesus': 9544,
        'chimp': 9598,
        'human': 9606,
        'dog': 9615,
        'cow': 9913,
    }
    assert species_name in species_to_homologene_id.keys()
    df = df[df['tax_id'] == species_to_homologene_id[species_name]]

    # Create a new transcript dict with new gene names
    name_to_id = {row["gene_symbol"]: row["hid"] for index, row in df.iterrows()}
    new_transcripts = list()
    for transcript in refseq_data.transcripts:
        # if transcript gene is in name to id dict then change the gene name to new name
        if transcript.gene in name_to_id.keys():
            transcript.gene = name_to_id[transcript.gene]

        new_transcripts.append(transcript)
    return RefseqDataset(new_transcripts)


def construct_human_mouse_transcript_dict(
    transcript_length_drop=12288,
    refseq_location_human="../data/gencode_basic_v41.tsv",
    refseq_location_mouse="../data/wgEncodeGencodeBaseicVM25.tsv",
    fasta_file_location_human="../data/hg38.fa",
    fasta_file_location_mouse="../data/mm10.fa",
    mini_dataset=False,
    do_homolog_map=False,
    drop_non_nm=False,
):
    # make sure that at least one of those is present
    assert bool(refseq_location_human) + bool(refseq_location_mouse) >= 1
    all_transcripts = []
    if refseq_location_human:
        refseq_data_human = prepare_refseq_dataset(
            transcript_length_drop=transcript_length_drop,
            refseq_location=refseq_location_human,
            fasta_file_location=fasta_file_location_human,
            mini_dataset=mini_dataset,
            drop_non_nm=drop_non_nm,
        )
        if do_homolog_map:
            refseq_data_human = rename_gene_names_using_homology_map(refseq_data_human)

        all_transcripts.extend(refseq_data_human.transcripts)

    if refseq_location_mouse:
        refseq_data_mouse = prepare_refseq_dataset(
            transcript_length_drop=transcript_length_drop,
            refseq_location=refseq_location_mouse,
            fasta_file_location=fasta_file_location_mouse,
            mini_dataset=mini_dataset,
            drop_non_nm=drop_non_nm,
        )
        if do_homolog_map:
            refseq_data_mouse = rename_gene_names_using_homology_map(refseq_data_mouse)

        all_transcripts.extend(refseq_data_mouse.transcripts)

    refseq_data = RefseqDataset(all_transcripts)

    transcript_dict = generate_gene_to_transcript_dict(refseq_data)
    return transcript_dict


def construct_combination_transcript_dict(
    refseq_files: tuple,
    fasta_files: tuple,
    species_names: tuple,
    transcript_length_drop=12288,
    mini_dataset=False,
    do_homolog_map=False,
    drop_non_nm=False,
    do_homologene_map=False,
):
    assert len(refseq_files) > 1
    assert len(fasta_files) > 1
    assert len(fasta_files) == len(refseq_files)
    # make sure that at least one of those is present
    all_transcripts = []
    for refseq_path, fasta_path, species_name in zip(refseq_files, fasta_files, species_names):
        print(species_name)

        # Valid chromosomes just implies chromosomes from chr1-chr22
        if species_name in ['human', 'mouse']:
            valid_chromosomes = True
        else:
            valid_chromosomes = False

        refseq_data = prepare_refseq_dataset(
            transcript_length_drop=transcript_length_drop,
            refseq_location=refseq_path,
            fasta_file_location=fasta_path,
            mini_dataset=mini_dataset,
            drop_non_nm=drop_non_nm,
            valid_chromosomes=valid_chromosomes,
        )
        if do_homolog_map:
            refseq_data = multi_species_homology_map(refseq_data, species_name)

        if do_homologene_map:
            refseq_data = multi_species_homologene_map(refseq_data, species_name)

        all_transcripts.extend(refseq_data.transcripts)

    refseq_data = RefseqDataset(all_transcripts)

    transcript_dict = generate_gene_to_transcript_dict(refseq_data)
    return transcript_dict


def write_tf_record_gene_pair_dataset_multi_genome(
    refseq_files: tuple,
    fasta_files: tuple,
    species_names: tuple,
    dataset_path,
    transcript_length_drop=12288,
    mini_dataset=False,
    zero_mean=False,
    zero_pad=True,
    n_genes_per_sub_dict=300,
    compression_type="ZLIB",
    drop_non_nm=False,
    drop_single_t_genes=False,
    n_tracks=6,
    do_homolog_map=False,
    do_homologene_map=False,
):
    # basic summary stats
    n_genes = 0
    n_transcripts = 0
    n_transcripts_per_gene_total = list()

    transcript_dict = construct_combination_transcript_dict(
        refseq_files=refseq_files,
        fasta_files=fasta_files,
        species_names=species_names,
        transcript_length_drop=transcript_length_drop,
        mini_dataset=mini_dataset,
        do_homolog_map=do_homolog_map,
        do_homologene_map=do_homologene_map,
        drop_non_nm=drop_non_nm,
    )

    # break up single transcript dict into multiple which will be written to each individual TFR file
    transcript_dicts = batch_transcript_dict(
        transcript_dict,
        n_genes_per_sub_dict=n_genes_per_sub_dict,
        drop_single_t_genes=drop_single_t_genes,
    )
    len(transcript_dicts)

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    tf_opts = tf.io.TFRecordOptions(compression_type=compression_type)

    for t_dict_index, transcript_dict in tqdm(
        enumerate(transcript_dicts), total=len(transcript_dicts), desc="# of TFR"
    ):
        n_genes += len(transcript_dict.keys())
        n_transcripts_per_gene_batch = [len(x) for x in transcript_dict.values()]
        n_transcripts_per_gene_total.extend(n_transcripts_per_gene_batch)
        n_transcripts += sum(n_transcripts_per_gene_batch)

        # Currently has identical pairs
        t_dataset, gene_id, n_transcripts_per_gene = construct_gene_np(
            transcript_dict,
            zero_mean=zero_mean,
            zero_pad=zero_pad,
            n_tracks=n_tracks,
        )

        tfrecord_file_name = f"{t_dict_index}.tfr"
        tfrecord_file_path = f"{dataset_path}/{tfrecord_file_name}"

        file_writer = tf.io.TFRecordWriter(tfrecord_file_path, tf_opts)
        feature = {
            "t_dataset": _floats_feature(t_dataset),
            "gene_id": _ints_feature(gene_id.astype(int)),
            "n_transcripts_per_gene": _ints_feature(n_transcripts_per_gene.astype(int)),
        }
        # encode transcripts
        record_bytes = tf.train.Example(
            features=tf.train.Features(feature=feature)
        ).SerializeToString()

        file_writer.write(record_bytes)

    with open(f"{dataset_path}_summary.txt", "w") as text_out:
        text_out.write(f"Number of genes: {n_genes}\n")
        text_out.write(f"Number of transcripts: {n_transcripts}\n\n")
        n_transcripts_per_gene_total = pd.Series(n_transcripts_per_gene_total)
        text_out.write(
            "Number of transcripts per gene:"
            f" \n{n_transcripts_per_gene_total.describe()}\n"
        )

    return dataset_path


def write_tf_record_gene_pair_dataset(
    dataset_path,
    transcript_length_drop=12288,
    refseq_location_human="",
    refseq_location_mouse="",
    fasta_file_location_human="../data/hg38.fa",
    fasta_file_location_mouse="../data/mm10.fa",
    mini_dataset=False,
    zero_mean=False,
    zero_pad=True,
    n_genes_per_sub_dict=150,
    compression_type="ZLIB",
    drop_non_nm=False,
    drop_single_t_genes=False,
    n_tracks=4,
    do_homolog_map=False,
):
    # basic summary stats
    n_genes = 0
    n_transcripts = 0
    n_transcripts_per_gene_total = list()

    transcript_dict = construct_human_mouse_transcript_dict(
        transcript_length_drop=transcript_length_drop,
        refseq_location_human=refseq_location_human,
        refseq_location_mouse=refseq_location_mouse,
        fasta_file_location_human=fasta_file_location_human,
        fasta_file_location_mouse=fasta_file_location_mouse,
        mini_dataset=mini_dataset,
        do_homolog_map=do_homolog_map,
        drop_non_nm=drop_non_nm,
    )
    # break up single transcript dict into multiple which will be written to each individual TFR file
    transcript_dicts = batch_transcript_dict(
        transcript_dict,
        n_genes_per_sub_dict=n_genes_per_sub_dict,
        drop_single_t_genes=drop_single_t_genes,
    )
    len(transcript_dicts)

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    tf_opts = tf.io.TFRecordOptions(compression_type=compression_type)

    for t_dict_index, transcript_dict in tqdm(
        enumerate(transcript_dicts), total=len(transcript_dicts), desc="# of TFR"
    ):
        n_genes += len(transcript_dict.keys())
        n_transcripts_per_gene_batch = [len(x) for x in transcript_dict.values()]
        n_transcripts_per_gene_total.extend(n_transcripts_per_gene_batch)
        n_transcripts += sum(n_transcripts_per_gene_batch)

        # Currently has identical pairs
        t_dataset, gene_id, n_transcripts_per_gene = construct_gene_np(
            transcript_dict,
            zero_mean=zero_mean,
            zero_pad=zero_pad,
            n_tracks=n_tracks,
        )

        tfrecord_file_name = f"{t_dict_index}.tfr"
        tfrecord_file_path = f"{dataset_path}/{tfrecord_file_name}"

        file_writer = tf.io.TFRecordWriter(tfrecord_file_path, tf_opts)
        feature = {
            "t_dataset": _floats_feature(t_dataset),
            "gene_id": _ints_feature(gene_id.astype(int)),
            "n_transcripts_per_gene": _ints_feature(n_transcripts_per_gene.astype(int)),
        }
        # encode transcripts
        record_bytes = tf.train.Example(
            features=tf.train.Features(feature=feature)
        ).SerializeToString()

        file_writer.write(record_bytes)

    with open(f"{dataset_path}_summary.txt", "w") as text_out:
        text_out.write(f"Number of genes: {n_genes}\n")
        text_out.write(f"Number of transcripts: {n_transcripts}\n\n")
        n_transcripts_per_gene_total = pd.Series(n_transcripts_per_gene_total)
        text_out.write(
            "Number of transcripts per gene:"
            f" \n{n_transcripts_per_gene_total.describe()}\n"
        )

    return dataset_path


def load_tf_record_gene_pair_dataset(
    dataset_path,
    transcript_length_drop=12288,
    compression_type="ZLIB",
    num_parallel_reads=8,
    interleave=True,
    n_tracks=4,
):
    dataset_path = os.path.expandvars(dataset_path)
    tfr_files = natsorted(glob.glob(f"{dataset_path}/*.tfr"))
    assert tfr_files

    def _parse_function(example_proto):
        keys_to_features = {
            "t_dataset": tf.io.RaggedFeature(tf.float32),
            "gene_id": tf.io.RaggedFeature(tf.int64),
            "n_transcripts_per_gene": tf.io.RaggedFeature(tf.int64),
        }
        parsed_features = tf.io.parse_single_example(example_proto, keys_to_features)

        ragged_tensors = tf.RaggedTensor.from_value_rowids(
            values=tf.reshape(parsed_features["t_dataset"], (-1, 12288, n_tracks)),
            value_rowids=parsed_features["gene_id"],
        )
        n_transcripts_per_gene = tf.cast(
            parsed_features["n_transcripts_per_gene"], tf.int32
        )
        return ragged_tensors, n_transcripts_per_gene

    def file_to_records(filename, compression_type="ZLIB"):
        dataset = tf.data.TFRecordDataset(filename, compression_type=compression_type)
        dataset = dataset.map(
            _parse_function,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return dataset

    if interleave:
        def uncompress_fn(x):
            return file_to_records(x, compression_type)

        dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
        parsed_dataset = dataset.interleave(
            map_func=uncompress_fn,
            cycle_length=4,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

    else:
        dataset = tf.data.TFRecordDataset(
            tfr_files,
            compression_type=compression_type,
            num_parallel_reads=num_parallel_reads,
        )
        parsed_dataset = dataset.map(
            _parse_function, num_parallel_calls=tf.data.AUTOTUNE
        )

    return parsed_dataset
