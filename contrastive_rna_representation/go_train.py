import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pandas as pd
import requests
import wandb
import os
from sklearn.model_selection import train_test_split
from absl import app
from absl import flags

from contrastive_rna_representation.resnet import (
    dilated_small,
    dilated_tiny,
    dilated_medium,
    dilated_small2,
    not_dilated_small,
    dilated_extra_tiny,
    dilated_small_constant_filters,
    dilated_large,
    less_dilated_small2,
)
from contrastive_rna_representation.contrastive_model import make_or_restore_model
from contrastive_rna_representation.gene_dataset import construct_human_mouse_transcript_dict
from contrastive_rna_representation.util import load_appris


def load_go_terms():
    # ### Read in the gene ontology table with name to GO terms mapping
    go_dir=(
        "/h/phil/Documents/01_projects/rna_half_life_branch/"
        "contrastive_rna_representation/data/gene_ontology.csv"
    )
    godf = pd.read_csv(go_dir)
    return godf


# ### Generate Root level GO terms
def get_root_children_go_terms(go_tree='mf', response=None):
    root_term_dict = {
        'mf': 'GO:0003674',  # molecular_function
        'bp': 'GO:0008150',  # biological_process
        'cc': 'GO:0005575'  # cellular_component
    }
    if not response:
        url = 'http://purl.obolibrary.org/obo/go/go-basic.obo'
        response = requests.get(url)
    if response.status_code == 200:
        data = response.text
        terms = data.split('\n\n')
        root_id = root_term_dict[go_tree]
        root_children = []
        for term in terms:
            if 'id:' in term:
                term_id = term.split('id: ')[1].rstrip()
                if term_id == root_id:
                    continue
                else:
                    parents = term.split('is_a: ')
                    for parent in parents[1:]:
                        parent_id = parent.split(' ! ')[0]
                        if parent_id == root_id:
                            go_name = term.split('name: ')[1].split('\n')[0]
                            go_id = term.split('id: ')[1].split('\n')[0]
                            root_children.append((go_id, go_name))
                            break
        return root_children
    else:
        return None


def get_two_levels_below_root_go_terms(go_tree='mf', response=None):
    root_term_dict = {
        'mf': 'GO:0003674',  # molecular_function
        'bp': 'GO:0008150',  # biological_process
        'cc': 'GO:0005575'  #c ellular_component
    }
    if not response:
        url = 'http://purl.obolibrary.org/obo/go/go-basic.obo'
        response = requests.get(url)

    if response.status_code == 200:
        data = response.text
        terms = data.split('\n\n')
        root_id = root_term_dict[go_tree]
        level_below_root = get_root_children_go_terms(go_tree, response)
        level_below_root_ids = [x[0] for x in level_below_root]
        two_levels_below_root_terms = []
        for term in terms:
            if 'id:' in term:
                term_id = term.split('id: ')[1].rstrip()
                if term_id == root_id:
                    continue
                elif term_id in level_below_root_ids:
                    continue
                else:
                    parents = term.split('is_a: ')
                    for parent in parents[1:]:
                        parent_id = parent.split(' ! ')[0]
                        if parent_id in level_below_root_ids:
                            go_name = term.split('name: ')[1].split('\n')[0]
                            go_id = term.split('id: ')[1].split('\n')[0]

                            two_levels_below_root_terms.append((go_id, go_name))
                            break
        return two_levels_below_root_terms
    else:
        return None


def get_three_levels_below_root_go_terms(go_tree='mf'):
    url = 'http://purl.obolibrary.org/obo/go/go-basic.obo'
    root_term_dict = {
        'mf': 'GO:0003674',  # molecular_function
        'bp': 'GO:0008150',  # biological_process
        'cc': 'GO:0005575'   #cellular_component
    }
    response = requests.get(url)
    if response.status_code == 200:
        data = response.text
        terms = data.split('\n\n')
        root_id = root_term_dict[go_tree]
        level_below_root = get_two_levels_below_root_go_terms(go_tree, response)
        level_below_root_ids = [x[0] for x in level_below_root]
        two_levels_below_root_terms = []
        for term in terms:
            if 'id:' in term:
                term_id = term.split('id: ')[1].rstrip()
                if term_id == root_id:
                    continue
                elif term_id in level_below_root_ids:
                    continue
                else:
                    parents = term.split('is_a: ')
                    for parent in parents[1:]:
                        parent_id = parent.split(' ! ')[0]
                        if parent_id in level_below_root_ids:
                            go_name = term.split('name: ')[1].split('\n')[0]
                            go_id = term.split('id: ')[1].split('\n')[0]

                            two_levels_below_root_terms.append((go_id, go_name))
                            break
        return two_levels_below_root_terms
    else:
        return None


def load_human_gene_transcript_map():
    # ### Make dataset
    dir = '/h/phil/Documents/01_projects/contrastive_rna_representation/annotation_data/'
    # generate only human t_map
    map = construct_human_mouse_transcript_dict(
        transcript_length_drop=12288,
        # refseq_location_human=f"{dir}/refseq_gencode_files/ncbi_refseq_curated_20221003.tsv",
        refseq_location_human=f"{dir}/refseq_gencode_files/human_comprehensive_gencode_v41_hg38.tsv",
        # refseq_location_mouse=f"{dir}/data/wgEncodeGencodeBaseicVM25.tsv",
        refseq_location_mouse="",
        fasta_file_location_human=f"{dir}/ref_genomes/hg38.fa",
        fasta_file_location_mouse=f"{dir}/ref_genomes/mm10.fa",
        mini_dataset=False,
        do_homolog_map=False,
        drop_non_nm=False,
    )
    n_genes = len(map.keys())
    n_transcripts = len(
        [transcript for t_list in map.values() for transcript in t_list]
    )
    print(f"Number of genes: {n_genes}")
    print(f"Number of transcripts: {n_transcripts}")
    return map


def generate_transcript_dataset_and_labels(df, map, go_map, args, n_tracks=6):
    pad_length_to = 12288
    zero_mean = False
    zero_pad = True

    assert 'gene_name' in df.columns
    assert 'ids' in df.columns
    assert 'Transcript ID' in df.columns

    # subset map to only columns in DF
    df['in_map'] = [x in map.keys() for x in df['gene_name']]
    df = df[df['in_map']]

    # subset df to only columns in map. (It's a merge)
    set_of_genes = set(df['gene_name'])
    map = {
        gene: transcripts for gene, transcripts in map.items() if gene in set_of_genes
    }

    # count total number of transcripts for creating of the array
    number_of_transcripts = sum([len(x) for x in map.values()])

    # create the arrays
    t_dataset = np.zeros(
        (number_of_transcripts, pad_length_to, n_tracks), dtype=np.float32
    )
    labels = np.zeros((number_of_transcripts, len(go_map.keys())), dtype=np.float32)

    i = 0
    for index, row in tqdm(df.iterrows(), total=len(df)):
        transcripts = map[row['gene_name']]

        for transcript in transcripts:
            if (
                args['single_transcript'] and
                transcript.transcript_id.split('.')[0] != row['Transcript ID']
            ):
                continue

            t_dataset[
                i, :, 0:4
            ] = transcript.one_hot_encode_transcript(
                pad_length_to=pad_length_to, zero_mean=zero_mean, zero_pad=zero_pad
            )
            if n_tracks >= 5:
                t_dataset[
                    i, :, 4:5
                ] = transcript.encode_coding_sequence_track(pad_length_to=pad_length_to)
            if n_tracks >= 6:
                t_dataset[
                    i, :, 5:6
                ] = transcript.encode_splice_track(pad_length_to=pad_length_to)

            for go_term in row['ids']:
                go_index = go_map[go_term]
                labels[i, go_index] = 1

            # iterate counter
            i += 1

    # Find the location of where the first empty row is and take that as the end index
    if args['single_transcript']:
        first_index_of_0 = np.min(np.where(t_dataset.sum((1,2)) == 0))
        t_dataset = t_dataset[:first_index_of_0]
        labels = labels[:first_index_of_0]
    print(t_dataset.shape)
    return t_dataset, labels


def write_aa_fasta_file(df, map, go_map, args):
    assert 'gene_name' in df.columns
    assert 'ids' in df.columns
    assert 'Transcript ID' in df.columns

    # subset map to only columns in DF
    df['in_map'] = [x in map.keys() for x in df['gene_name']]
    df = df[df['in_map']]

    # subset df to only columns in map. (It's a merge)
    set_of_genes = set(df['gene_name'])
    map = {
        gene: transcripts for gene, transcripts in map.items() if gene in set_of_genes
    }

    fasta_string = ""

    for index, row in tqdm(df.iterrows(), total=len(df)):
        transcripts = map[row['gene_name']]

        for transcript in transcripts:
            if (
                args['single_transcript'] and
                transcript.transcript_id.split('.')[0] != row['Transcript ID']
            ):
                continue
            if transcript.cds_start - transcript.cds_end == 0:
                continue

            aa_seq = transcript.get_amino_acid_sequence()
            fasta_label = f"{transcript.gene}_{transcript.transcript_id}"

            fasta_string += f">{fasta_label}\n"
            fasta_string += aa_seq + '\n'
    # Find the location of where the first empty row is and take that as the end index
    return fasta_string


def create_rna_sequence_numpy_file(df, map, go_map, args):
    pad_length_to = 12288

    assert 'gene_name' in df.columns
    assert 'ids' in df.columns
    assert 'Transcript ID' in df.columns

    # subset map to only columns in DF
    df['in_map'] = [x in map.keys() for x in df['gene_name']]
    df = df[df['in_map']]

    # subset df to only columns in map. (It's a merge)
    set_of_genes = set(df['gene_name'])
    map = {
        gene: transcripts for gene, transcripts in map.items() if gene in set_of_genes
    }
    number_of_transcripts = sum([len(x) for x in map.values()])

    t_dataset = np.chararray((number_of_transcripts, pad_length_to))
    labels = np.zeros((number_of_transcripts, len(go_map.keys())), dtype=np.float32)

    i = 0
    for index, row in tqdm(df.iterrows(), total=len(df)):
        transcripts = map[row['gene_name']]

        for transcript in transcripts:
            if (
                args['single_transcript'] and
                transcript.transcript_id.split('.')[0] != row['Transcript ID']
            ):
                continue

            t_dataset[i] = list(transcript.get_sequence(pad_length_to=pad_length_to))
            for go_term in row['ids']:
                go_index = go_map[go_term]
                labels[i, go_index] = 1

            # iterate counter
            i += 1

    # Find the location of where the first empty row is and take that as the end index
    if args['single_transcript']:
        t_dataset = t_dataset[:i]
        labels = labels[:i]
    print(t_dataset.shape)
    return t_dataset, labels


def create_go_tf_dataset(args, write_fasta=False, write_rna_sequence=False):
    high_evidence=args['high_evidence']
    number_of_go_terms = args['num_classes']

    three_below_root = get_three_levels_below_root_go_terms('mf')
    godf = load_go_terms()
    app_h = load_appris(unique_transcripts=args['single_transcript'])
    map = load_human_gene_transcript_map()

    print(f"Number of unique genes: {len(godf['gene_name'].unique())}")
    # subset to root level go terms
    gene_root_go = godf[godf['id'].isin([x[0] for x in three_below_root])]
    print(f"Number of genes with root level GO {gene_root_go['gene_name'].nunique()}")

    # Drop different levels of evidence
    gene_root_go = gene_root_go[~gene_root_go[['gene_name', 'id']].duplicated()]

    # Subset by evidence or not
    if high_evidence:
        gene_root_go = gene_root_go[
            gene_root_go['evidence'].isin(['IDA', 'IMP', 'IGI', 'IPI', 'TAS'])
        ]

    # subset to top X go terms
    top_X_go_terms = gene_root_go['id'].value_counts()[:number_of_go_terms].keys()
    gene_root_go = gene_root_go[gene_root_go['id'].isin(top_X_go_terms)]

    # for every row assign a list of go terms
    gene_go = gene_root_go.groupby(['gene_name'], as_index=False).agg({'id': list})
    gene_go = gene_go.rename(columns={'id': 'ids'})

    # merge with appris to generate single transcript per gene
    gene_go = app_h.merge(
        gene_go, left_on='Gene name', right_on='gene_name'
    )[['gene_name', 'Transcript ID', 'ids']]

    print(len(gene_go))
    # t_data, samples = generate_transcript_dataset_and_labels(gene_root_go, map)

    # create a map of go ids to enum
    all_go_terms = [id for ids in gene_go['ids'].values for id in ids]
    go_map = {go_id: i for i, go_id in enumerate(set(all_go_terms))}

    if write_fasta:
        fasta_go = write_aa_fasta_file(gene_go, map, go_map, args)
        with open('../data/aa_go_all_t_fasta.fa', 'w') as fin:
            fin.write(fasta_go)

    if write_rna_sequence:
        t_dataset, labels = create_rna_sequence_numpy_file(gene_go, map, go_map, args)
        np.savez(
            '../data/go_data/go_dna_dataset',
            t_dataset=t_dataset,
            labels=labels,
        )

    index = gene_go.index
    train_index, test_index = train_test_split(
        index, test_size=0.2, random_state=args['rand_seed']
    )
    train_index, val_index = train_test_split(
        train_index, test_size=0.2, random_state=args['rand_seed']
    )

    train_dataset, train_labels = generate_transcript_dataset_and_labels(
        gene_go.loc[train_index], map, go_map, args, n_tracks=args['n_tracks']
    )

    test_dataset, test_labels = generate_transcript_dataset_and_labels(
        gene_go.loc[test_index], map, go_map, args, n_tracks=args['n_tracks']
    )

    val_dataset, val_labels = generate_transcript_dataset_and_labels(
        gene_go.loc[val_index], map, go_map, args, n_tracks=args['n_tracks']
    )

    print(
        "Shape of input sequences and the corresponding labels",
        train_labels.shape,
        train_dataset.shape
    )
    print('Number of GO terms per sample')
    print(pd.Series(train_labels.sum(axis=1)).value_counts())

    assert test_dataset.sum() != 0
    assert train_dataset.sum() != 0
    assert val_dataset.sum() != 0

    train_data = tf.data.Dataset.from_tensor_slices((train_dataset, train_labels))
    test_data = tf.data.Dataset.from_tensor_slices((test_dataset, test_labels))
    val_data = tf.data.Dataset.from_tensor_slices((val_dataset, val_labels))
    train_data = train_data.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
    test_data = test_data.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
    val_data = val_data.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
    train_data = train_data.batch(args['global_batch_size'])
    test_data = test_data.batch(args['global_batch_size'])
    val_data = val_data.batch(args['global_batch_size'])
    return train_data, test_data, val_data


def make_optimizer(args):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        args["lr"],
        decay_steps=10_000 / args["global_batch_size"] * 2,
        decay_rate=.95,
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        global_clipnorm=args['clipnorm'],
        amsgrad=False,
        beta_1=0.90,
        beta_2=0.998,
    )
    return optimizer


def compile_model(model, args):
    sample = np.random.randn(10, 12288, args["n_tracks"])
    model.supervised_inference(sample)
    model(sample)
    model.contrastive_inference(sample)
    model.projection_head = None

    model_metrics = [
        tf.keras.metrics.AUC(curve='ROC'),
        tf.keras.metrics.AUC(curve='PR')
    ]
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    optimizer = make_optimizer(args)
    model.compile(
        loss=loss_fn, optimizer=optimizer, metrics=model_metrics
    )
    model.build(input_shape=(None, 12288, args["n_tracks"]))
    model.compute_output_shape(input_shape=(None, 12288, args["n_tracks"]))
    print(model.summary())


def load_weights_from_contrastive_model(
    model, contrastive_model, n_tracks, load_pool_weights
):
    sample = np.random.randn(10, 12288, n_tracks)
    model.supervised_inference(sample)
    model(sample)
    model.body.set_weights(contrastive_model.model.body.get_weights())

    if load_pool_weights:
        model.avgpool.set_weights(contrastive_model.model.avgpool.get_weights())


def load_model(args):
    resnet_model = None

    if args["resnet"] == "dilated_small":
        resnet_model = dilated_small

    elif args["resnet"] == "dilated_small2":
        resnet_model = dilated_small2

    elif args["resnet"] == "dilated_large":
        resnet_model = dilated_large

    elif args["resnet"] == "less_dilated_small2":
        resnet_model = less_dilated_small2

    elif args["resnet"] == "dilated_medium":
        resnet_model = dilated_medium

    elif args["resnet"] == "dilated_tiny":
        resnet_model = dilated_tiny

    elif args["resnet"] == "dilated_extra_tiny":
        resnet_model = dilated_extra_tiny

    elif args["resnet"] == "dilated_small_constant_filters":
        resnet_model = dilated_small_constant_filters

    elif args["resnet"] == "not_dilated_small":
        resnet_model = not_dilated_small
        
    else:
        raise ValueError

    model = resnet_model(
        args["num_classes"],
        dropout_prob=args["dropout_prob"],
        norm_type=args["norm_type"],
        l2_scale=args["l2_scale_weight_decay"],
        kernel_initializer=args["kernel_initializer"],
        pooling_layer=args["pooling_layer"],
        kernel_size=args["kernel_size"],
        fc_head=args["fc_head"],
        max_pool=args['max_pool'],
    )

    # if contrastive model
    if args["contrastive_run_dir"]:
        # which args should I be using here
        # Should probably load them
        checkpoint_dir = f"{args['contrastive_run_dir']}/checkpoints"
        assert os.path.isdir(checkpoint_dir)
        contrast_model, contrast_epoch = make_or_restore_model(
            checkpoint_dir,
            {
                "lr": 1,
                "hl_lr": 1,
                "mixed_precision": False,
                "weight_decay": 0,
                'l2_scale_weight_decay': args['l2_scale_weight_decay'],
                'clipnorm': args['clipnorm'],
            },
            checkpoint_epoch=args["contrastive_checkpoint_epoch"],
        )
        assert contrast_epoch != 0
        # set conv body to the contrast model conv body
        print(
            f"Loading weights from {args['contrastive_run_dir']}_epoch"
            f" {contrast_epoch}"
        )
        load_weights_from_contrastive_model(
            model, contrast_model, args["n_tracks"], args["load_pool_weights"]
        )
    model.body.trainable = args['train_full_model']
    compile_model(model, args)
    return model


def final_evaluation(model, test_dataset, dataset_prefix=''):
    result = model.evaluate(test_dataset)
    result_dict = dict(zip(model.metrics_names, result))
    result_dict = {
        dataset_prefix + key: value for key,value in result_dict.items()
    }
    return result_dict


def init_wandb(args, project='train_go'):

    if args["contrastive_run_dir"]:
        args["from_contrastive"] = True
    else:
        args["from_contrastive"] = False

    frozen = not args['train_full_model']
    run_id = (
        f"{args['note']}-pre_{args['from_contrastive']}-frz_{frozen}-"
        # f"1t_{args['single_transcript']}-highe_{args['high_evidence']}-ngo_{args['num_classes']}"
        f"{args['resnet']}-lr_{args['lr']}-l2wd_{args['l2_scale_weight_decay']}-"
        f"d_{args['dropout_prob']}-seed_{args['rand_seed']}-"
        f"bs_{args['global_batch_size']}"
        # f"-frac_{args['fraction_of_train']}"
    )
    print('RUN ID:', len(run_id), run_id)
    id_no_seed = (
        f"{args['note']}-pre_{args['from_contrastive']}-frz_{frozen}-"
        # f"1t_{args['single_transcript']}-ngo_{args['num_classes']}-"
        f"{args['resnet']}-lr_{args['lr']}-l2wd_{args['l2_scale_weight_decay']}-"
        f"d_{args['dropout_prob']}-"
        f"bs_{args['global_batch_size']}"
        # f"-frac_{args['fraction_of_train']}"
    )

    # define and make path
    wandb_id_file_path_str = f"{args['wandb_run_dir']}/{project}-{run_id}"
    if not os.path.isdir(wandb_id_file_path_str):
        os.makedirs(wandb_id_file_path_str)

    args["id_no_seed"] = id_no_seed

    wandb.init(
        project=project,
        config=args,
        id=run_id,
        dir=wandb_id_file_path_str,
    )


def main(argv):
    args = FLAGS.flag_values_dict()
    tf.keras.utils.set_random_seed(args['rand_seed'])
    print(args)
    # create dataset
    train_data, test_data, val_data = create_go_tf_dataset(args)
    model = load_model(args)

    init_wandb(args)

    # create callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
    wadnb_cb = wandb.keras.WandbCallback(
        monitor="val_loss",
        mode="min",
        save_model=False,
    )

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=args['number_of_epochs'],
        callbacks=[
            wadnb_cb,
            early_stop
        ],
        verbose=1,
    )

    final_test_results = final_evaluation(model, test_data, dataset_prefix='final_test_')
    final_val_results = final_evaluation(model, val_data, dataset_prefix='final_val_')
    wandb.log(final_test_results)
    wandb.log(final_val_results)


if __name__ == "__main__":
    FLAGS = flags.FLAGS

    flags.DEFINE_bool("mixed_precision", False, "Whether to use mixed precision")
    flags.DEFINE_integer("rand_seed", 42, "random seed")
    flags.DEFINE_string("note", "", "note for wandb")
    flags.DEFINE_string(
        "wandb_run_dir",
        "/scratch/hdd001/home/phil/rna_contrast/runs",
        "Wandb run directory",
    )

    flags.DEFINE_bool('single_transcript', True, 'use single transcript per gene')
    flags.DEFINE_bool('high_evidence', False, 'Subset to GO terms that have high evidence')
    flags.DEFINE_integer('num_classes', 10, 'Number of GO terms to use')
    flags.DEFINE_integer(
        "n_tracks", 6, "number of data tracks that the model is trained on"
    )
    flags.DEFINE_integer(
        "input_shape", 12288, "Length of the input sequence",
    )
    flags.DEFINE_float(
        "fraction_of_train",
        1.0,
        "Fraction of training_data to subsample",
    )

    # MODEL STUFF
    flags.DEFINE_bool('train_full_model', True, 'Whether to update the weights of body')
    # contrastive model stuff
    flags.DEFINE_string(
        "contrastive_run_dir",
        "",
        "Directory from which to load the model. If missing initializes new one",
    )
    flags.DEFINE_bool(
        "load_pool_weights",
        False,
        "Whether to load contrastive model weights for pooling layer",
    )
    flags.DEFINE_integer(
        "contrastive_checkpoint_epoch",
        0,
        "If set to 0 loads the latest epoch. Else provide the epoch from which to load",
    )
    flags.DEFINE_string("resnet", "dilated_small", "Type of model to use")
    flags.DEFINE_float("dropout_prob", 0.3, "dropout probability")
    flags.DEFINE_string("norm_type", "batchnorm_small_momentum", "Type of normalization to use")
    flags.DEFINE_integer("kernel_size", 2, "size of kernel to use for convolutions")
    flags.DEFINE_string("pooling_layer", "avgpool", "What kind of layer to use to pool")
    flags.DEFINE_float("l2_scale_weight_decay", 0, "l2 regularization for weights")
    flags.DEFINE_string("kernel_initializer", "he_normal", "What kind of initialization to use")
    flags.DEFINE_string("fc_head", "linear_sigmoid", "Type of fully connected head to use")
    flags.DEFINE_string("max_pool", 'max_pool', "What kind of pooling layer to use")

    # optimizer
    flags.DEFINE_float("lr", 1e-3, "hl fine tune learning rate")
    flags.DEFINE_float("clipnorm", 5.0, "Clip the norm of the gradients beyond that")
    flags.DEFINE_integer("number_of_epochs", 100, "Number of epochs")
    flags.DEFINE_float("weight_decay", 0, "Decay weights by this value every epoch")
    flags.DEFINE_integer("global_batch_size", 128, "Number of samples per batch")

    app.run(main)
