import tensorflow as tf
from wandb.keras import WandbCallback
import logging
import os
from absl import app
from absl import flags
from tensorflow.keras import mixed_precision
import tensorflow_addons as tfa

from contrastive_rna_representation.resnet import (
    dilated_small,
    dilated_tiny,
    dilated_medium,
    dilated_extra_tiny,
    not_dilated_small,
    dilated_small2,
    dilated_large,
    not_dilated_medium,
    less_dilated_small,
    less_dilated_small2,
    less_dilated_medium,
    less_dilated_medium2,
)
from contrastive_rna_representation.util import (
    load_gene_pair_dataset_from_tf_records,
    load_gene_pair_dataset_from_tf_records_w_t_count,
    make_timestamp,
)
from contrastive_rna_representation.saluki_layers import SalukiModel
from contrastive_rna_representation.util import (
    init_wandb,
    LRLogger,
    ReciprocalSqrtLearningRateScheduler,
    ReciprocalSqrtLearningRateSchedulerBatch,
)

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
LARGE_NUM=1e9
SMALL_NUM = 1e-9

# Define the contrastive model with model-subclassing
class ContrastiveModel(tf.keras.Model):
    def __init__(
        self,
        number_devices,
        resnet,
        dropout_prob,
        global_batch_size,
        temperature=0.1,
        norm_type="batchnorm",
        dilation_params=(),
        kernel_size=2,
        l2_scale_weight_decay=0,
        projection_head_size=128,
        projection_body=512,
        pooling_layer="avgpool",
        loss_name='NTXentLoss_2',
        mixed_precision=True,
        train_weighted="",
        gradient_checkpointing=False,
        subtract_sim_max=False,
    ):
        super().__init__()
        self.number_devices = number_devices
        self.temperature = temperature
        # self.temperature = tf.Variable(initial_value=temperature, trainable=True, dtype=tf.float32)

        self.resnet = resnet
        self.dropout_prob = dropout_prob
        self.global_batch_size = global_batch_size
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.l2_scale_weight_decay = l2_scale_weight_decay
        self.projection_head_size = projection_head_size
        self.projection_body = projection_body
        self.pooling_layer = pooling_layer
        self.loss_name = loss_name
        self.train_weighted = train_weighted
        self.mixed_precision = mixed_precision
        self.gradient_checkpointing = gradient_checkpointing
        self.subtract_sim_max = subtract_sim_max

        if self.loss_name == "NTXentLoss_2":
            self.loss_fn = self.contrastive_loss_2
        elif self.loss_name == 'DCL':
            self.loss_fn = self.dcl_loss
        else:
            raise ValueError

        resnet_model = None
        if self.resnet == "dilated_small":
            resnet_model = dilated_small

        elif self.resnet == "dilated_small2":
            resnet_model = dilated_small2

        elif self.resnet == "less_dilated_small":
            resnet_model = less_dilated_small

        elif self.resnet == "less_dilated_small2":
            resnet_model = less_dilated_small2

        elif self.resnet == "dilated_medium":
            resnet_model = dilated_medium

        elif self.resnet == "less_dilated_medium":
            resnet_model = less_dilated_medium

        elif self.resnet == "less_dilated_medium2":
            resnet_model = less_dilated_medium2

        elif self.resnet == "dilated_tiny":
            resnet_model = dilated_tiny

        elif self.resnet == "dilated_extra_tiny":
            resnet_model = dilated_extra_tiny

        elif self.resnet == "not_dilated_small":
            resnet_model = not_dilated_small

        elif self.resnet == "dilated_large":
            resnet_model = dilated_large

        elif self.resnet == "not_dilated_medium":
            resnet_model = not_dilated_medium

        elif self.resnet == "saluki":
            self.model = SalukiModel()

        if resnet_model:
            self.model = resnet_model(
                1,
                dropout_prob=self.dropout_prob,
                norm_type=norm_type,
                l2_scale=l2_scale_weight_decay,
                kernel_initializer="he_normal",
                kernel_size=kernel_size,
                projection_head_size=projection_head_size,
                pooling_layer=pooling_layer,
                projection_body=projection_body,
            )

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)

        self.optimizer = optimizer

        self.contrastive_loss_tracker = tf.keras.metrics.Mean(name="contrastive_loss")
        self.loss_positive_term = tf.keras.metrics.Mean(name="positive_term")
        self.loss_negative_term = tf.keras.metrics.Mean(name="negative_term")
        self.norm_max = tf.keras.metrics.Mean(name='norm_max')
        self.norm_min = tf.keras.metrics.Mean(name='norm_min')
        self.similarity_mean = tf.keras.metrics.Mean(name='similarity_mean')
        self.positive_loss_max = tf.keras.metrics.Mean(name='positive_max')
        self.positive_loss_min = tf.keras.metrics.Mean(name='positive_min')
        self.negative_loss_min_00 = tf.keras.metrics.Mean(name='negative_loss_min_00')
        self.negative_loss_max_00 = tf.keras.metrics.Mean(name='negative_loss_max_00')
        self.negative_loss_min_01 = tf.keras.metrics.Mean(name='negative_loss_min_01')
        self.negative_loss_max_01 = tf.keras.metrics.Mean(name='negative_loss_max_01')
        self.last_layer_max_weight = tf.keras.metrics.Mean(name='last_layer_max_weight')
        self.last_layer_min_weight = tf.keras.metrics.Mean(name='last_layer_min_weight')
        self.projection_min = tf.keras.metrics.Mean(name='projection_min')
        self.projection_max = tf.keras.metrics.Mean(name='projection_max')
        self.contrastive_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="contrastive_accuracy"
        )

    @property
    def metrics(self):
        return [
            self.contrastive_loss_tracker,
            self.contrastive_accuracy,
            self.loss_positive_term,
            self.loss_negative_term,
            self.norm_max,
            self.norm_min,
            self.similarity_mean,
            self.positive_loss_max,
            self.positive_loss_min,
            self.negative_loss_min_00,
            self.negative_loss_max_00,
            self.negative_loss_min_01,
            self.negative_loss_max_01,
            self.last_layer_max_weight,
            self.last_layer_min_weight,
            self.projection_min,
            self.projection_max,
        ]

    def contrastive_loss_2(self, projections_1, projections_2, sample_weight):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # Cosine similarity: the dot product of the l2-normalized feature vectors
        proj_1 = tf.math.l2_normalize(projections_1, axis=1)
        proj_2 = tf.math.l2_normalize(projections_2, axis=1)

        batch_size = tf.shape(projections_1)[0]

        if self.number_devices > 1:
            proj_1_all = tf.distribute.get_replica_context().all_gather(proj_1, axis=0)
            proj_2_all = tf.distribute.get_replica_context().all_gather(proj_2, axis=0)
            replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group
            enlarged_batch_size = tf.shape(proj_1_all)[0]
            labels_idx = tf.range(batch_size) + replica_id * batch_size
            labels = tf.one_hot(labels_idx, enlarged_batch_size * 2)
            masks = tf.one_hot(labels_idx, enlarged_batch_size)
        else:
            proj_1_all = proj_1
            proj_2_all = proj_2
            labels_idx = tf.range(batch_size)
            labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
            masks = tf.one_hot(tf.range(batch_size), batch_size)

        logits_aa = tf.matmul(proj_1, proj_1_all, transpose_b=True) / self.temperature
        logits_aa = logits_aa - tf.cast(masks * LARGE_NUM, tf.float32)
        logits_bb = tf.matmul(proj_2, proj_2_all, transpose_b=True) / self.temperature
        logits_bb = logits_bb - tf.cast(masks * LARGE_NUM, tf.float32)
        logits_ab = tf.matmul(proj_1, proj_2_all, transpose_b=True) / self.temperature
        logits_ba = tf.matmul(proj_2, proj_1_all, transpose_b=True) / self.temperature

        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        self.contrastive_accuracy.update_state(labels_idx, tf.concat([logits_ab, logits_aa], 1))
        self.contrastive_accuracy.update_state(labels_idx, tf.concat([logits_ba, logits_bb], 1))

        loss_a = tf.nn.softmax_cross_entropy_with_logits(
            labels, tf.concat([logits_ab, logits_aa], 1)
        )
        loss_b = tf.nn.softmax_cross_entropy_with_logits(
            labels, tf.concat([logits_ba, logits_bb], 1)
        )

        return tf.math.reduce_mean((loss_a + loss_b) * sample_weight)

    def dcl_loss(self, projections_1, projections_2, sample_weight):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        self.norm_min.update_state(tf.reduce_min(tf.norm(projections_1, axis=1)))
        self.norm_max.update_state(tf.reduce_max(tf.norm(projections_1, axis=1)))

        self.projection_max.update_state(tf.reduce_min(projections_1))
        self.projection_min.update_state(tf.reduce_max(projections_1))

        # Cosine similarity: the dot product of the l2-normalized feature vectors
        proj_1 = tf.math.l2_normalize(projections_1 + SMALL_NUM, axis=1)
        proj_2 = tf.math.l2_normalize(projections_2 + SMALL_NUM, axis=1)

        if self.number_devices > 1:
            proj_1_all = tf.distribute.get_replica_context().all_gather(proj_1, axis=0)
            proj_2_all = tf.distribute.get_replica_context().all_gather(proj_2, axis=0)
        else:
            proj_1_all = proj_1
            proj_2_all = proj_2

        loss_1_2 = self._dcl_loss(proj_1, proj_2, proj_1_all, proj_2_all, sample_weight)
        loss_2_1 = self._dcl_loss(proj_2, proj_1, proj_2_all, proj_1_all, sample_weight)

        return (loss_1_2 + loss_2_1) / 2

    def _dcl_loss(self, proj_1, proj_2, proj_1_all, proj_2_all, sample_weight):
        # get batch size and replica id
        batch_size = tf.shape(proj_1)[0]
        enlarged_batch_size = tf.shape(proj_1_all)[0]
        replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group

        labels_idx = tf.range(batch_size) + replica_id * batch_size
        diag_mask = tf.one_hot(labels_idx, enlarged_batch_size)
        diag_mask = tf.cast(diag_mask, tf.bool)

        sim_00 = tf.einsum("nc,mc->nm", proj_1, proj_1_all) / self.temperature
        sim_01 = tf.einsum("nc,mc->nm", proj_1, proj_2_all) / self.temperature
        if self.subtract_sim_max:
            sim_00 = sim_00 - tf.math.reduce_max(sim_00)
            sim_01 = sim_01 - tf.math.reduce_max(sim_01)

        self.similarity_mean.update_state(tf.math.reduce_mean(sim_00))
        # update accuracy before removing the positives from the matrix
        self.contrastive_accuracy.update_state(labels_idx, sim_01)

        positive_loss = -sim_01[diag_mask] * sample_weight
        self.positive_loss_max.update_state(tf.reduce_max(positive_loss))
        self.positive_loss_min.update_state(tf.reduce_min(positive_loss))

        sim_00 = tf.reshape(sim_00[~diag_mask], (batch_size, -1))
        sim_01 = tf.reshape(sim_01[~diag_mask], (batch_size, -1))

        negative_loss_00 = tf.math.reduce_logsumexp(sim_00, axis=1)
        negative_loss_01 = tf.math.reduce_logsumexp(sim_01, axis=1)

        self.negative_loss_min_01.update_state(tf.reduce_min(negative_loss_01))
        self.negative_loss_min_00.update_state(tf.reduce_min(negative_loss_00))

        self.negative_loss_max_00.update_state(tf.reduce_max(negative_loss_00))
        self.negative_loss_max_01.update_state(tf.reduce_max(negative_loss_01))

        positive_term = tf.math.reduce_mean(positive_loss)
        negative_term = tf.math.reduce_mean(negative_loss_00 + negative_loss_01)

        self.loss_positive_term.update_state(positive_term)
        self.loss_negative_term.update_state(negative_term)

        return positive_term + negative_term

    def call(self, data):
        hl_pred = self.model.supervised_inference(data)
        projection = self.model.contrastive_inference(data, training=False)
        representation = self.model(data)

        return hl_pred, projection, representation

    def train_step(
        self,
        data,
    ):
        self.last_layer_max_weight.update_state(
            tf.reduce_max(self.model.projection_head.layers[-1].weights)
        )
        self.last_layer_min_weight.update_state(
            tf.reduce_min(self.model.projection_head.layers[-1].weights)
        )

        if 'by_transcript' in self.train_weighted:
            t1, t2, sample_weight = data
        else:
            t1, t2 = data
            sample_weight = tf.constant(1., tf.float32)

        with tf.GradientTape() as tape:
            projections_1 = self.model.contrastive_inference(t1, training=True)
            projections_2 = self.model.contrastive_inference(t2, training=True)

            contrast_loss = self.loss_fn(projections_1, projections_2, sample_weight)
            contrast_loss = contrast_loss + tf.cast(
                sum(self.model.losses), tf.float32
            )
            # mixed precision acrobatics
            if self.mixed_precision:
                scaled_contrast_loss = self.optimizer.get_scaled_loss(
                    contrast_loss
                )

        # mixed precision acrobatics
        if self.mixed_precision:
            scaled_gradients = tape.gradient(
                scaled_contrast_loss,
                self.model.body.trainable_weights
                + self.model.projection_head.trainable_weights
            )
            gradients = self.optimizer.get_unscaled_gradients(
                scaled_gradients
            )

        else:
            gradients = tape.gradient(
                contrast_loss,
                self.model.body.trainable_weights
                + self.model.projection_head.trainable_weights
            )
        # apply gradients
        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.model.body.trainable_weights
                + self.model.projection_head.trainable_weights
            )
        )
        self.contrastive_loss_tracker.update_state(contrast_loss)

        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        config = super().get_config()

        config["number_devices"] = self.number_devices
        config["resnet"] = self.resnet
        config["dropout_prob"] = self.dropout_prob
        config["global_batch_size"] = self.global_batch_size
        # Gather temperature values from all replicas
        config['temperature'] = self.temperature
        config["norm_type"] = self.norm_type
        config["kernel_size"] = self.kernel_size
        config["l2_scale_weight_decay"] = self.l2_scale_weight_decay
        config["projection_head_size"] = self.projection_head_size
        config["pooling_layer"] = self.pooling_layer
        config['projection_body'] = self.projection_body
        config["loss_name"] = self.loss_name
        config["mixed_precision"] = self.mixed_precision
        config["train_weighted"] = self.train_weighted
        config["gradient_checkpointing"] = self.gradient_checkpointing
        config['subtract_sim_max'] = self.subtract_sim_max

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def custom_pop(sequential_object, layer_type):
    """_summary_
    Sometimes the model adds an additional Input layer at the end of the layer stack after loading.
    This creates challenges when training in mixed precision as the output of the model is supposed
    to be tf.float32. Instead by default it gets initialized to tf.float16

    To circumvent this I pop the last layer of the model when its type is  tf.keras.layers.InputLayer.

    Args:
        sequential_object (tf.keras.Sequential): A sequential object
        layer_type (tf.keras.layers): A tf.keras Layer
    """
    new_tracked_trackables = [
        x for x in sequential_object._self_tracked_trackables[1:] if type(x) != layer_type
    ]
    new_tracked_trackables = [sequential_object._self_tracked_trackables[0]] + new_tracked_trackables
    layer_to_remove = [
        x for x in sequential_object._self_tracked_trackables[1:] if type(x) == layer_type
    ]
    assert len(layer_to_remove) == 1
    layer_to_remove = layer_to_remove[0]
    sequential_object._layer_call_argspecs.pop(layer_to_remove)
    sequential_object._self_tracked_trackables = new_tracked_trackables


def make_or_restore_model(checkpoint_dir, args, checkpoint_epoch=None):
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.

    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]

    if checkpoints:
        if checkpoint_epoch:
            checkpoint_epochs = [int(x.split("-")[-1]) for x in checkpoints]
            epoch_ckpt_tuple = zip(checkpoint_epochs, checkpoints)
            required_checkpoint = [
                x[1] for x in epoch_ckpt_tuple if x[0] == checkpoint_epoch
            ]
            assert len(required_checkpoint) == 1
            checkpoint = required_checkpoint[0]

        else:
            # load the latest checkpoint
            checkpoint = max(checkpoints, key=os.path.getctime)

        logging.info(f"Restoring from {checkpoint}")
        current_epoch = int(checkpoint.split("-")[-1])
        model = tf.keras.models.load_model(
            checkpoint,
            compile=False,
            custom_objects={
                "ContrastiveModel": ContrastiveModel,
            },
        )
        # Change model precision
        settings_to_check = [
            'mixed_precision',
            'l2_scale_weight_decay',
            'norm_type',
            'dropout_prob',
            "train_weighted",
        ]
        for setting in settings_to_check:
            if setting in args.keys() and model.__dict__[setting] != args[setting]:
                print(
                    f"Warning: Changing {setting} from "
                    f"{model.__dict__[setting]} to {args[setting]}"
                )
                model.__dict__[setting] = args[setting]


        # for some reason keras upon saving adds a InputLayer at the end of the sequential models
        # This doesn't affect the forward pass besides changing the output to float16
        # during mixed precision training
        print(len(model.model.fc.layers), model.model.fc.layers[-1])

        if type(model.model.fc.layers[-1]) == tf.keras.layers.InputLayer:
            # loaded_model.model.fc.pop()
            custom_pop(model.model.fc, tf.keras.layers.InputLayer)

        if type(model.model.projection_head.layers[-1]) == tf.keras.layers.InputLayer:
            # loaded_model.model.projection_head.pop()
            custom_pop(model.model.projection_head, tf.keras.layers.InputLayer)

        print(len(model.model.fc.layers), model.model.fc.layers[-1])

        if 'optimizer' not in args.keys() or args['optimizer'] == 'adam':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=args["lr"],
                weight_decay=args["weight_decay"],
                clipnorm=args['clipnorm'],
            )
        elif args['optimizer'] == 'adamw':
            optimizer = tfa.optimizers.AdamW(
                learning_rate=args["lr"],
                weight_decay=args["weight_decay"],
                clipnorm=args['clipnorm']
            )

        elif args['optimizer'] == 'lamb':
            optimizer = tfa.optimizers.LAMB(
                learning_rate=args["lr"],
                weight_decay=args["weight_decay"],
                clipnorm=args['clipnorm']
            )
        else:
            raise ValueError

        if args["mixed_precision"]:
            # Already set
            optimizer = mixed_precision.LossScaleOptimizer(
                optimizer
            )

        model.compile(optimizer=optimizer)
        return model, current_epoch

    logging.info("Creating a new model")
    return get_compiled_model(args), 0


def get_compiled_model(args):
    if args['resnet'] == 'dilated_large':
        gradient_checkpointing = True
    else:
        gradient_checkpointing = False

    model = ContrastiveModel(
        number_devices=args["number_devices"],
        resnet=args["resnet"],
        dropout_prob=args["dropout_prob"],
        global_batch_size=args["global_batch_size"],
        norm_type=args["norm_type"],
        l2_scale_weight_decay=args["l2_scale_weight_decay"],
        temperature=args["temperature"],
        kernel_size=args["kernel_size"],
        projection_head_size=args["projection_head_size"],
        projection_body=args['projection_body'],
        pooling_layer=args["pooling_layer"],
        loss_name=args['contrastive_loss_name'],
        mixed_precision=args['mixed_precision'],
        train_weighted=args['train_weighted'],
        gradient_checkpointing=gradient_checkpointing,
        subtract_sim_max=args['subtract_sim_max'],
    )

    if args['optimizer'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=args["lr"],
            weight_decay=args["weight_decay"],
            clipnorm=args['clipnorm']
        )
    elif args['optimizer'] == 'adamw':
        optimizer = tfa.optimizers.AdamW(
            learning_rate=args["lr"],
            weight_decay=args["weight_decay"],
            clipnorm=args['clipnorm']
        )
    elif args['optimizer'] == 'lamb':
        optimizer = tfa.optimizers.LAMB(
            learning_rate=args["lr"],
            weight_decay=args["weight_decay"],
            clipnorm=args['clipnorm']
        )
    else:
        raise ValueError

    if args['mixed_precision']:
        optimizer = mixed_precision.LossScaleOptimizer(
            optimizer
        )

    model.compile(
        optimizer=optimizer,
    )
    return model


def main(argv):
    print(f"The time is now {make_timestamp()}")
    print("FLAGS.debug is {}".format(FLAGS.debug))
    print("FLAGS.mixed_precision is {}".format(FLAGS.mixed_precision))
    args = FLAGS.flag_values_dict()

    if args["mixed_precision"]:
        mixed_precision.set_global_policy('mixed_float16')

    print(f"Replica: {tf.distribute.get_replica_context().replica_id_in_sync_group}")
    print(args)

    tf.random.set_seed(args["rand_seed"])

    # Set strategy
    strategy = tf.distribute.MirroredStrategy()
    options = tf.data.Options()
    # Each worker gets their own file
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.FILE
    )

    number_devices = strategy.num_replicas_in_sync
    logging.info(f"Number of devices: {number_devices}.")
    args["number_devices"] = number_devices

    run, run_dir = init_wandb(args, FLAGS)

    # If dataset path is set use an existing TFRecord dataset
    if args["dataset_path"]:
        if args["train_weighted"] == 'by_transcript':
            train_data = load_gene_pair_dataset_from_tf_records_w_t_count(args)
        elif args["train_weighted"] == 'by_transcript_0.6':
            dataset_name = args['dataset_path'].split('/')[-1]
            assert dataset_name in ['11_genome_gene_pair', '10_genome_homologene']
            train_data = load_gene_pair_dataset_from_tf_records_w_t_count(
                args,
                normalize_ratio=0.908,
                additive_constant=0.6
            )
        elif args["train_weighted"] == 'by_transcript_0.4':
            dataset_name = args['dataset_path'].split('/')[-1]
            assert dataset_name in ['11_genome_gene_pair', '10_genome_homologene', '10_genome_homologene_4_track', '10_genome_no_homologene']
            train_data = load_gene_pair_dataset_from_tf_records_w_t_count(
                args,
                normalize_ratio=0.802,
                additive_constant=0.4
            )

        elif args["train_weighted"] == 'by_transcript_1.0':
            dataset_name = args['dataset_path'].split('/')[-1]
            assert dataset_name in ['11_genome_gene_pair', '10_genome_homologene']
            train_data = load_gene_pair_dataset_from_tf_records_w_t_count(
                args,
                normalize_ratio=1.08,
                additive_constant=1.0
            )
        else:
            train_data = load_gene_pair_dataset_from_tf_records(args)
    # Load and store dataset in memory
    else:
        train_data = load_gene_pair_dataset_from_tf_records(args)

    # Create a checkpoint directory which will be used for saving models
    checkpoint_dir = f"{run_dir}/checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Open a strategy scope.
    with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.
        model, current_epoch = make_or_restore_model(
            checkpoint_dir, args, args['contrastive_checkpoint_epoch'],
        )
        model.compute_output_shape(input_shape=(None, 12288, args["n_tracks"]))

    #if args['dataset_path'].split('/')[-1] == "10_genome_homologene":
    #    dataset_size = 228_800
    #else:
    #    raise ValueError

    # lr_scheduler = ReciprocalSqrtLearningRateSchedulerBatch(
    #     dataset_size,
    #     args['global_batch_size'],
    #     args['number_epochs'],
    #     args['lr'],
    #     warmup_epochs=10,
    #     cooldown_epochs=20,
    #     min_lr=1e-8,
    #     verbose=0
    # )
    # lr_scheduler.set_current_epoch(current_epoch)
    lr_scheduler = ReciprocalSqrtLearningRateScheduler(
        args['number_epochs'],
        args['lr'],
        warmup_epochs=10,
        cooldown_epochs=50,
        min_lr=1e-8,
        verbose=0
    )

    model.fit(
        train_data,
        epochs=args["number_epochs"],
        initial_epoch=current_epoch,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=run_dir + "/checkpoints" + "/ckpt-{epoch}",
                mode="min",
                monitor="contrastive_loss",
                # save every 20 batches
                save_freq='epoch',
                period=4,
                # save_freq=30,
                save_best_only=False,
                verbose=1,
            ),
            lr_scheduler,
            LRLogger(model.optimizer),
            WandbCallback(
                monitor="contrastive_loss",
                mode="min",
                log_batch_frequency=50,
                save_model=False,
                log_weights=True,
            ),
        ],
    )


if __name__ == "__main__":
    # Misc flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string("note", "", "note for wandb")
    flags.DEFINE_float("dropout_prob", 0.1, "dropout probability")
    flags.DEFINE_integer("rand_seed", 0, "random seed")

    # Optimizer flags
    flags.DEFINE_float("lr", 1e-4, "contrastive learning rate")
    flags.DEFINE_float("clipnorm", 5.0, "Clip the norm of the gradients beyond that")
    flags.DEFINE_float("weight_decay", 0, "Decay weights by this value every epoch")
    flags.DEFINE_float("l2_scale_weight_decay", 0, "l2 regularization for weights")
    flags.DEFINE_string("optimizer", "adam", "Optimizer to use: adam or lamb")
    # Model flags
    flags.DEFINE_string("resnet", "resnet_34", "Type of model to use")
    flags.DEFINE_string("norm_type", "batchnorm", "Type of normalization to use")
    flags.DEFINE_integer("kernel_size", 3, "size of kernel to use for convolutions")
    flags.DEFINE_integer("projection_head_size", 128, "size of projection head")
    flags.DEFINE_integer("projection_body", 512, "size of layers leading up to projection head")
    flags.DEFINE_string("pooling_layer", "avgpool", "What kind of layer to use to pool")
    flags.DEFINE_float("temperature", 0.1, "temperature used for cosine similarity normalization")
    # training flags
    flags.DEFINE_string('contrastive_loss_name', 'NTXentLoss_2', 'which loss to use')
    flags.DEFINE_string('train_weighted', '', 'Whether to weight the loss by transcript')
    flags.DEFINE_integer(
        "contrastive_checkpoint_epoch",
        0,
        "If set to 0 loads the latest epoch. Else provide the epoch from which to load",
    )
    flags.DEFINE_bool(
        "subtract_sim_max", False, "whether to subtract max from similarity matrix",
    )


    # Data flags
    flags.DEFINE_integer("global_batch_size", 256, "Total batch size")
    flags.DEFINE_string("compression_type", "ZLIB", "Type of compression used while writing the TFRecord file")
    flags.DEFINE_integer("number_epochs", 100, "Number of epochs")
    flags.DEFINE_integer("samples", 18229, "Number of samples")
    flags.DEFINE_float(
        "proportion_to_mask", 0.0, "proportion of the transcript to mask as N"
    )
    flags.DEFINE_integer(
        "n_tracks", 4, "number of data tracks that the model is trained on"
    )
    flags.DEFINE_bool(
        "mask_single_transcript",
        False,
        "whether to mask just a single transcript or both",
    )

    flags.DEFINE_bool(
        "zero_mean", False, "Whether one hot encoded sequences have 0 mean"
    )
    flags.DEFINE_bool(
        "same_transcript", False, "Whether to always select the same transcript"
    )
    flags.DEFINE_bool(
        "always_different_transcripts",
        False,
        "Whether to always select different transcript (when more than 1)"
    )
    flags.DEFINE_bool(
        "zero_pad", True, "Whether to zero pad the end of the sequence instead of 0.25"
    )
    flags.DEFINE_string(
        "wandb_run_dir",
        "/scratch/hdd001/home/phil/rna_contrast/runs",
        "Wandb run directory",
    )
    flags.DEFINE_string(
        "dataset_path",
        "$HOME/Documents/01_projects/contrastive_rna_representation/data/gene_pair_dataset_refseq",
        "Directory of TFRecord File",
    )
    flags.DEFINE_bool("mixed_precision", False, "Whether to use mixed precision")

    app.run(main)
    # Contrastive pretraining
