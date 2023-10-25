import wandb
import logging
import os
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp
import numpy as np
from datetime import datetime
import re
import dataclasses
from typing import List
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import _result_classes, pearsonr, spearmanr
import math
import pandas as pd
from tqdm import tqdm

from contrastive_rna_representation.gene_dataset import construct_human_mouse_transcript_dict
from contrastive_rna_representation.gene_dataset import (
    load_tf_record_gene_pair_dataset,
)


def load_transcript_map(mouse=False):
    dir = '/h/phil/Documents/01_projects/contrastive_rna_representation/annotation_data/'
    # generate only human t_map
    if mouse:
        mouse_location = f"{dir}/refseq_gencode_files/mouse_comprehensive_gencodevm25_mm10.tsv"
    else:
        mouse_location = ""
    map = construct_human_mouse_transcript_dict(
        transcript_length_drop=12288,
        refseq_location_human=f"{dir}/refseq_gencode_files/human_comprehensive_gencode_v41_hg38.tsv",
        refseq_location_mouse=mouse_location,
        fasta_file_location_human=f"{dir}/ref_genomes/hg38.fa",
        fasta_file_location_mouse=f"{dir}/ref_genomes/mm10.fa",
        mini_dataset=False,
        do_homolog_map=False,
        drop_non_nm=False,
    )
    return map


def load_appris(unique_transcripts=True):
    # ## load human appris
    dir = '/h/phil/Documents/01_projects/contrastive_rna_representation/'

    app_h = pd.read_csv(f'{dir}/data/appris_data_human.principal.txt', sep='\t')
    print(app_h['Gene ID'].duplicated().sum())
    app_h['numeric_value'] = app_h['APPRIS Annotation'].str.split(':').str[1]
    app_h['key_value'] = app_h['APPRIS Annotation'].str.split(':').str[0]
    app_h = app_h.sort_values(
        ['Gene ID', 'key_value','numeric_value', "Transcript ID"],
        ascending=[True, False, True, True],
    )
    if unique_transcripts:
        app_h = app_h[~app_h.duplicated('Gene ID')]
        app_h = app_h[~app_h.duplicated('Gene name')]
    return app_h


class CosineLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, max_epochs, initial_lr, warmup_epochs=0, verbose=0):
        super(CosineLearningRateScheduler, self).__init__()
        self.max_epochs = max_epochs
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.initial_lr * (1 + math.cos(math.pi * progress)) / 2

        if self.verbose > 0:
            print(f"Epoch {epoch+1}: Learning rate = {lr:.6f}")
        self.model.optimizer.lr.assign(lr)


class ReciprocalSqrtLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, max_epochs, initial_lr, warmup_epochs=0, cooldown_epochs=0, min_lr=1e-8, verbose=0):
        super(ReciprocalSqrtLearningRateScheduler, self).__init__()
        self.max_epochs = max_epochs
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.cooldown_epochs = cooldown_epochs
        self.min_lr = min_lr  # Minimum learning rate during cooldown
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        elif epoch >= self.max_epochs - self.cooldown_epochs:
            # Gradually reduce learning rate during the cooldown period
            cooldown_progress = (epoch - (self.max_epochs - self.cooldown_epochs)) / self.cooldown_epochs
            lr = (1 - cooldown_progress) * self.initial_lr / (1 + epoch) ** 0.5 + cooldown_progress * self.min_lr
        else:
            lr = self.initial_lr / (1 + epoch - self.warmup_epochs) ** 0.5

        if self.verbose > 0:
            print(f"Epoch {epoch+1}: Learning rate = {lr:.6f}")
        self.model.optimizer.lr.assign(lr)


class ReciprocalSqrtLearningRateSchedulerBatch(tf.keras.callbacks.Callback):
    def __init__(
        self,
        dataset_size,
        batch_size,
        total_epochs,
        initial_lr,
        warmup_epochs=0,
        cooldown_epochs=0,
        min_lr=1e-8,
        verbose=0
    ):
        super(ReciprocalSqrtLearningRateSchedulerBatch, self).__init__()
        self.steps_per_epoch = dataset_size // batch_size
        self.total_steps = self.steps_per_epoch * total_epochs
        self.warmup_steps = self.steps_per_epoch * warmup_epochs
        self.cooldown_steps = self.steps_per_epoch * cooldown_epochs
        self.initial_lr = initial_lr
        self.min_lr = min_lr  # Minimum learning rate during cooldown
        self.verbose = verbose
        self.global_step = 0

    def set_current_epoch(self, current_epoch):
        """Set the global_step value based on the current epoch."""
        self.global_step = current_epoch * self.steps_per_epoch

    def on_batch_begin(self, batch, logs=None):
        if self.global_step < self.warmup_steps:
            lr = self.initial_lr * (self.global_step + 1) / self.warmup_steps
        elif self.global_step >= self.total_steps - self.cooldown_steps:
            # Gradually reduce learning rate during the cooldown period
            cooldown_progress = (
                self.global_step - (self.total_steps - self.cooldown_steps)
                ) / self.cooldown_steps
            lr = (
                (1 - cooldown_progress) * self.initial_lr /
                (1 + self.global_step) ** 0.5 +
                cooldown_progress * self.min_lr
            )
        else:
            lr = self.initial_lr / (1 + self.global_step - self.warmup_steps) ** 0.5

        if self.verbose > 0:
            print(f"Step {self.global_step+1}: Learning rate = {lr:.6f}")
        self.model.optimizer.lr.assign(lr)
        self.global_step += 1


class PearsonR(tf.keras.metrics.Metric):
    def __init__(self, num_targets=1, summarize=True, name="pearsonr", **kwargs):
        super(PearsonR, self).__init__(name=name, **kwargs)
        self._summarize = summarize
        self._shape = (num_targets,)
        self._count = self.add_weight(
            name="count", shape=self._shape, initializer="zeros"
        )

        self._product = self.add_weight(
            name="product", shape=self._shape, initializer="zeros"
        )
        self._true_sum = self.add_weight(
            name="true_sum", shape=self._shape, initializer="zeros"
        )
        self._true_sumsq = self.add_weight(
            name="true_sumsq", shape=self._shape, initializer="zeros"
        )
        self._pred_sum = self.add_weight(
            name="pred_sum", shape=self._shape, initializer="zeros"
        )
        self._pred_sumsq = self.add_weight(
            name="pred_sumsq", shape=self._shape, initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, "float32")
        y_pred = tf.cast(y_pred, "float32")

        if len(y_true.shape) == 2:
            reduce_axes = 0
        else:
            reduce_axes = [0, 1]

        product = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=reduce_axes)
        self._product.assign_add(product)

        true_sum = tf.reduce_sum(y_true, axis=reduce_axes)
        self._true_sum.assign_add(true_sum)

        true_sumsq = tf.reduce_sum(tf.math.square(y_true), axis=reduce_axes)
        self._true_sumsq.assign_add(true_sumsq)

        pred_sum = tf.reduce_sum(y_pred, axis=reduce_axes)
        self._pred_sum.assign_add(pred_sum)

        pred_sumsq = tf.reduce_sum(tf.math.square(y_pred), axis=reduce_axes)
        self._pred_sumsq.assign_add(pred_sumsq)

        count = tf.ones_like(y_true)
        count = tf.reduce_sum(count, axis=reduce_axes)
        self._count.assign_add(count)

    def result(self):
        true_mean = tf.divide(self._true_sum, self._count)
        true_mean2 = tf.math.square(true_mean)
        pred_mean = tf.divide(self._pred_sum, self._count)
        pred_mean2 = tf.math.square(pred_mean)

        term1 = self._product
        term2 = -tf.multiply(true_mean, self._pred_sum)
        term3 = -tf.multiply(pred_mean, self._true_sum)
        term4 = tf.multiply(self._count, tf.multiply(true_mean, pred_mean))
        covariance = term1 + term2 + term3 + term4

        true_var = self._true_sumsq - tf.multiply(self._count, true_mean2)
        pred_var = self._pred_sumsq - tf.multiply(self._count, pred_mean2)
        pred_var = tf.where(
            tf.greater(pred_var, 1e-12), pred_var, np.inf * tf.ones_like(pred_var)
        )

        tp_var = tf.multiply(tf.math.sqrt(true_var), tf.math.sqrt(pred_var))
        correlation = tf.divide(covariance, tp_var)

        if self._summarize:
            return tf.reduce_mean(correlation)
        else:
            return correlation

    def reset_states(self):
        K.batch_set_value([(v, np.zeros(self._shape)) for v in self.variables])


def init_wandb(args, FLAGS):
    project = "rna_contrast_5"

    # create a unique ID for the run
    run_id = (
        f"{args['note']}-pool_{args['pooling_layer']}-{args['contrastive_loss_name']}"
        f"dev_{args['number_devices']}-d_{args['dropout_prob']}-"
        f"seed_{args['rand_seed']}-bs_{args['global_batch_size']}-"
        f"{args['resnet']}-lr_{args['lr']}-e_{args['number_epochs']}"
    )
    id_no_seed = (
        f"{args['note']}-pool_{args['pooling_layer']}-{args['contrastive_loss_name']}"
        f"dev_{args['number_devices']}-d_{args['dropout_prob']}-"
        f"bs_{args['global_batch_size']}-"
        f"{args['resnet']}-lr_{args['lr']}-e_{args['number_epochs']}"
    )

    wandb_id_file_path_str = f"{args['wandb_run_dir']}/{project}-{run_id}"
    args["id_no_seed"] = id_no_seed

    # if run id exists:
    if os.path.isdir(wandb_id_file_path_str):
        logging.info(f"{run_id} exists loading config")

        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        if slurm_job_id is not None:
            cache_dir = f'/checkpoint/{slurm_job_id}'
        else:
            cache_dir = wandb_id_file_path_str
        os.environ['WANDB_CACHE_DIR'] = cache_dir

        resume_id = run_id

        run = wandb.init(
            id=run_id,
            resume=resume_id,
            project=project,
            config=args,
            dir=wandb_id_file_path_str,
        )
        args = wandb.config

    else:
        logging.info(f"{run_id} doesn't exist writing to file")

        # Make a directory corresponding to run name
        os.makedirs(wandb_id_file_path_str)
        # save config file
        FLAGS.append_flags_into_file(f"{wandb_id_file_path_str}/config.json")

        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        if slurm_job_id is not None:
            cache_dir = f'/checkpoint/{slurm_job_id}'
        else:
            cache_dir = wandb_id_file_path_str
        os.environ['WANDB_CACHE_DIR'] = cache_dir
        # init Wandb
        run = wandb.init(
            id=run_id,
            project=project,
            config=args,
            dir=wandb_id_file_path_str,
        )
        args = wandb.config

    return run, wandb_id_file_path_str


def select_random_transcript(
    matrix_of_transcripts,
    n_transcripts,
    same_transcript=False,
    always_different_transcripts=False,
):
    probabilities = tf.ones(n_transcripts) / tf.cast(n_transcripts, tf.float32)

    distribution = tfp.distributions.OneHotCategorical(
        logits=None,
        probs=probabilities,
        dtype=tf.int32,
        validate_args=False,
        allow_nan_stats=True,
        name="OneHotCategorical",
    )
    index1 = tf.squeeze(tf.where(distribution.sample()))
    index2 = tf.squeeze(tf.where(distribution.sample()))

    # Sample new index until it doesn't equal the first one
    if always_different_transcripts and n_transcripts > 1:
        while index1 == index2:
            index2 = tf.squeeze(tf.where(distribution.sample()))

    transcript_1 = matrix_of_transcripts[index1, :, :]
    if not same_transcript:
        transcript_2 = matrix_of_transcripts[index2, :, :]
    else:
        transcript_2 = matrix_of_transcripts[index1, :, :]
    return transcript_1, transcript_2


def n_mask_proportion_of_transcript(
    transcript,
    proportion,
    t_length=12288,
    zero_mean=False,
    n_tracks=4,
):
    # number of positions to be masked
    num_mask = int(proportion * t_length)

    # construct a (t_length, 4) mask
    idxs = tf.range(t_length)
    idxs = tf.random.shuffle(idxs)[:num_mask]
    idxs = tf.reshape(idxs, (-1, 1))
    # construct an all False mask and set indexes to True
    bool_mask = tf.scatter_nd(
        idxs, tf.ones(num_mask, dtype=bool), tf.constant([t_length])
    )
    bool_mask = tf.reshape(bool_mask, (-1, 1))
    # Broadcast to (t_length, 4)
    bool_mask = tf.broadcast_to(bool_mask, (t_length, n_tracks))
    # replacement mask (N's are 0.25 padding is 0, ACGT ordering 1, 0, 0, 0 (A)) if not zero mean
    replacement = tf.zeros((t_length, n_tracks))
    if not zero_mean:
        replacement = replacement + 0.25
    transcript = tf.where(bool_mask, replacement, transcript)

    return transcript


def mask_two_transcripts_wrapper(
    t1, t2, proportion_to_mask, zero_mean, mask_single_transcript, n_tracks=4
):
    transcript_1 = n_mask_proportion_of_transcript(
        t1,
        proportion=proportion_to_mask,
        zero_mean=zero_mean,
        n_tracks=n_tracks,
    )

    if mask_single_transcript:
        transcript_2 = t2
    else:
        transcript_2 = n_mask_proportion_of_transcript(
            t2,
            proportion=proportion_to_mask,
            zero_mean=zero_mean,
            n_tracks=n_tracks,
        )
    return transcript_1, transcript_2


def load_gene_pair_dataset_from_tf_records(
    args,
):
    train_dataset = load_tf_record_gene_pair_dataset(
        dataset_path=args["dataset_path"],
        interleave=True,
        num_parallel_reads=args["number_devices"] * 4,
        compression_type=args["compression_type"],
        n_tracks=args["n_tracks"],
    )
    train_dataset = train_dataset.unbatch()
    train_dataset = train_dataset.map(
        lambda x, y: select_random_transcript(
            x,
            y,
            same_transcript=args['same_transcript'],
            always_different_transcripts=args['always_different_transcripts']
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if args["proportion_to_mask"]:
        train_dataset = train_dataset.map(
            lambda t1, t2: mask_two_transcripts_wrapper(
                t1,
                t2,
                args["proportion_to_mask"],
                args["zero_mean"],
                args["mask_single_transcript"],
                n_tracks=args["n_tracks"],
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    train_dataset = train_dataset.shuffle(
        buffer_size=10 * args["global_batch_size"], reshuffle_each_iteration=True
    )
    train_dataset = train_dataset.batch(
        args["global_batch_size"],
        drop_remainder=True,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    return train_dataset


def load_gene_pair_dataset_from_tf_records_w_t_count(
    args,
    normalize_ratio=None,
    additive_constant=0.2
):
    train_dataset = load_tf_record_gene_pair_dataset(
        dataset_path=args["dataset_path"],
        interleave=True,
        num_parallel_reads=args["number_devices"] * 4,
        compression_type=args["compression_type"],
        n_tracks=args["n_tracks"],
    )

    train_dataset = train_dataset.unbatch()
    train_dataset = train_dataset.map(
        lambda x, y:
            (*select_random_transcript(
                x,
                y,
                same_transcript=args['same_transcript'],
                always_different_transcripts=args['always_different_transcripts']
                ),
            y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # have to change normalization for different dataset
    # Want the overall norm of the loss to stay the same
    # So we calculate the number of transcripts per gene
    # Apply a log transformation to penalize small count less than large
    # TODO ugly hack change later

    # Two transcriptomes Human + mouse
    if not normalize_ratio:
        if args['dataset_path'].split('/')[-1] == (
            "gene_pair_human_mouse_6t_homolog_drop_single"
        ):
            normalize_ratio = tf.constant(1.944, tf.float32)
        elif args['dataset_path'].split('/')[-1] == (
            "gene_pair_human_mouse_6t_homolog"
        ):
            normalize_ratio = tf.constant(0.965, tf.float32)

        # 11 genomes using homologene map
        elif args['dataset_path'].split('/')[-1] == (
            "11_genome_gene_pair_drop_single"
        ):
            normalize_ratio = tf.constant(1.76, tf.float32)
        elif args['dataset_path'].split('/')[-1] == (
            "11_genome_gene_pair"
        ):
            normalize_ratio = tf.constant(0.68, tf.float32)

        # TODO calculate actual ratio
        elif args['dataset_path'].split('/')[-1] == (
            "10_genome_homologene"
        ):
            normalize_ratio = tf.constant(0.68, tf.float32)
        elif args['dataset_path'].split('/')[-1] == (
            "11_genome_gene_pair_no_homology"
        ):
            normalize_ratio = tf.constant(0.68, tf.float32)

        # Single genome Human
        elif args['dataset_path'].split('/')[-1] == (
            "gene_pair_human_6t_homolog_drop_single"
        ):
            normalize_ratio = tf.constant(1.88, tf.float32)
        elif args['dataset_path'].split('/')[-1] == (
            "gene_pair_human_6t_homolog"
        ):
            normalize_ratio = tf.constant(1.05, tf.float32)

        else:
            raise ValueError

    train_dataset = train_dataset.map(
        lambda t1, t2, y: (
            t1,
            t2,
            tf.math.divide_no_nan(
                tf.math.log(tf.cast(y, tf.float32) + additive_constant),
                normalize_ratio
            )
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if args["proportion_to_mask"]:
        train_dataset = train_dataset.map(
            lambda t1, t2, y: (* mask_two_transcripts_wrapper(
                t1,
                t2,
                args["proportion_to_mask"],
                args["zero_mean"],
                args["mask_single_transcript"],
                n_tracks=args["n_tracks"],
            ), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    train_dataset = train_dataset.shuffle(
        buffer_size=10 * args["global_batch_size"], reshuffle_each_iteration=True
    )
    train_dataset = train_dataset.batch(
        args["global_batch_size"],
        drop_remainder=True,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    return train_dataset


def make_timestamp():
    timestamp = "_".join(re.split(":|-| ", str(datetime.now()).split(".")[0]))
    return timestamp


class LRLogger(tf.keras.callbacks.Callback):
    def __init__(self, optimizer):
        super(LRLogger, self).__init__()
        self.optimizer = optimizer

    def on_batch_end(self, batch, logs=None):
        lr = self.optimizer.learning_rate.numpy()
        wandb.log({"lr": lr}, commit=False)


@dataclasses.dataclass
class ScatterplotResults:
    figure: Figure
    correlation: float
    p_value: float
    correlation_spearman: float
    pvalue_spearman: float
    mse: float
    # confidence_level: float
    # upper_correlation_bound: float
    # lower_correlation_bound: float


def plot_scatterplot_and_correlation(
    x: List[float], y: List[float], title: str, xlabel: str, ylabel: str
) -> ScatterplotResults:
    # clear seaborn
    sns.reset_orig()
    sns.set()
    sns.set_context("talk")

    f, axes = plt.subplots(1)
    # use seaborn style defaults and set the default figure size
    sns.set(rc={"figure.figsize": (8, 8)})
    # use x and y as the data, assign to the variables called x and y
    # use the function regplot to make a scatterplot
    # color the scatterplot points blue
    # pass axes so you don't overwrite the same plots
    plot = sns.regplot(x=x, y=y, color="b", line_kws={"color": "red"}, ax=axes)
    # add a (1, 1) line to show perfect correlation
    plot.plot([0, 1], [0, 1], transform=plot.transAxes, ls="--", c=".3")

    # Calculate the correlation coefficient between x and y
    pearson: _result_classes.PearsonRResult = pearsonr(
        x=x,
        y=y,
    )
    spearman: _result_classes.SpearmanRResult = spearmanr(
        x,
        y,
    )
    mse = np.mean((x - y)**2)

    # confidence_level = 0.95
    # confidence_interval: ConfidenceInterval = pearson.confidence_interval(
    #     confidence_level=confidence_level
    # )
    # lower_bound = confidence_interval[0]
    # upper_bound = confidence_interval[1]
    correlation: float = pearson[0]
    pvalue: float = pearson[1]

    correlation_spearman: float = spearman[0]
    pvalue_spearman: float = spearman[1]

    # set a title for the regplot
    title_with_statistics = f"{title} Correlation: {correlation:.2f} p {pvalue:.2f}"
    plot.figure.suptitle(title_with_statistics)
    # set the labels for the x and y axes
    plot.set(xlabel=xlabel, ylabel=ylabel)
    # set the x and y axis to (0, 1)
    # plot.set(xlim=(0, 1), ylim=(0, 1))
    figure = plot.figure

    return ScatterplotResults(
        figure=figure,
        correlation=correlation,
        p_value=pvalue,
        correlation_spearman=correlation_spearman,
        pvalue_spearman=pvalue_spearman,
        mse=mse,
        # confidence_level=confidence_level,
        # upper_correlation_bound=upper_bound,
        # lower_correlation_bound=lower_bound,
    )
