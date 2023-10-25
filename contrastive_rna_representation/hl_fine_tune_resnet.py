from absl import app
from absl import flags
from tensorflow.keras import mixed_precision
import tensorflow as tf
import numpy as np
import wandb
from wandb.keras import WandbCallback
import os
import json

from contrastive_rna_representation.util import load_hl_data, PearsonCorrelation
from contrastive_rna_representation.resnet import (
    resnet_34,
    resnet_50,
    resnet_18,
    resnet_101,
    resnet_152,
    dilated_small,
    dilated_medium,
    dilated_tiny,
)
from contrastive_rna_representation.contrastive_model import make_or_restore_model


class FineTuneModel(tf.keras.Model):
    def __init__(
        self,
        resnet,
        dropout_prob,
        global_hl_batch_size,
        num_classes,
        contrastive_checkpoint_epoch=0,
    ):
        super().__init__()
        self.resnet = resnet
        self.dropout_prob = dropout_prob
        self.global_hl_batch_size = global_hl_batch_size
        self.num_classes = num_classes
        self.contrastive_checkpoint_epoch = contrastive_checkpoint_epoch

        assert resnet in [
            "resnet_18",
            "resnet_34",
            "resnet_50",
            "resnet_101",
            "dilated_small",
            "dilated_medium",
            "dilated_tiny",
        ]

        if resnet == "resnet_18":
            self.model = resnet_18(self.num_classes, dropout_prob=self.dropout_prob)
        elif resnet == "resnet_34":
            self.model = resnet_34(self.num_classes, dropout_prob=self.dropout_prob)
        elif resnet == "resnet_50":
            self.model = resnet_50(self.num_classes, dropout_prob=self.dropout_prob)
        elif resnet == "resnet_101":
            self.model = resnet_101(self.num_classes, dropout_prob=self.dropout_prob)
        elif resnet == "resnet_152":
            self.model = resnet_152(self.num_classes, dropout_prob=self.dropout_prob)
        elif resnet == "dilated_small":
            self.model = dilated_small(self.num_classes, dropout_prob=self.dropout_prob)
        elif resnet == "dilated_medium":
            self.model = dilated_medium(
                self.num_classes, dropout_prob=self.dropout_prob
            )
        elif resnet == "dilated_tiny":
            self.model = dilated_tiny(self.num_classes, dropout_prob=self.dropout_prob)

        # Remove projection head since this is only the fine tuning model
        self.model.projection_head = None

        sample = np.random.randn(10, 12288, 4)
        self.model.supervised_inference(sample)
        self.model(sample)
        self.model.summary()

    def load_weights_from_contrastive_model(self, contrastive_model):
        assert self.resnet == contrastive_model.resnet
        assert self.dropout_prob == contrastive_model.dropout_prob
        self.model.body = contrastive_model.model.body

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)

        self.optimizer = optimizer

        self.fine_tune_loss = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )

        self.mouse_loss_tracker = tf.keras.metrics.Mean(name="mse_loss_half_life_mouse")
        self.human_loss_tracker = tf.keras.metrics.Mean(name="mse_loss_half_life_human")
        self.human_pearson = PearsonCorrelation(name="human_pearson")
        self.mouse_pearson = PearsonCorrelation(name="mouse_pearson")

    @property
    def metrics(self):
        return [
            self.mouse_loss_tracker,
            self.human_loss_tracker,
            self.human_pearson,
            self.mouse_pearson,
        ]

    def call(self, data):
        # dummy call method
        hl_pred = self.model.supervised_inference(data)
        representation = self.model(data)

        return hl_pred, representation

    def training_sub_step(self, seq, half_life, head_num):
        # Human loss
        with tf.GradientTape() as tape:
            # perform inference

            pred = self.model.supervised_inference(seq)[:, head_num : head_num + 1]
            # calculate loss
            # divide by batch size since reduction is sum
            # loss = self.fine_tune_loss(half_life, pred) / self.global_hl_batch_size
            loss = self.fine_tune_loss(half_life, pred)

            if FLAGS.mixed_precision:
                scaled_loss = self.optimizer.get_scaled_loss(loss)
        # mixed precision acrobatics

        if FLAGS.mixed_precision:
            scaled_gradients = tape.gradient(
                scaled_loss,
                self.model.body.trainable_weights + self.model.fc.trainable_weights,
            )
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(
                loss,
                self.model.body.trainable_weights + self.model.fc.trainable_weights,
            )
        # apply gradients
        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.model.body.trainable_weights + self.model.fc.trainable_weights,
            )
        )
        return loss, pred

    def train_step(
        self,
        datasets,
    ):
        human_data, mouse_data = datasets
        human_seq, human_hl = human_data
        mouse_seq, mouse_hl = mouse_data

        mouse_loss, mouse_pred = self.training_sub_step(mouse_seq, mouse_hl, 1)
        self.mouse_loss_tracker.update_state(mouse_loss)
        self.mouse_pearson.update_state(mouse_hl, mouse_pred)

        human_loss, human_pred = self.training_sub_step(human_seq, human_hl, 0)
        self.human_loss_tracker.update_state(human_loss)
        self.human_pearson.update_state(human_hl, human_pred)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, half_life_data):
        human_data, mouse_data = half_life_data
        human_seq, human_hl = human_data
        mouse_seq, mouse_hl = mouse_data

        # For testing the components are used with a training=False flag
        human_pred = self.model.supervised_inference(human_seq, training=False)[:, 0:1]
        mouse_pred = self.model.supervised_inference(mouse_seq, training=False)[:, 1:2]

        # have to do this since the reduction is sum. Divide by batch size and multiply by # devices
        human_loss = self.fine_tune_loss(human_hl, human_pred)
        mouse_loss = self.fine_tune_loss(mouse_hl, mouse_pred)

        # human_loss = (
        #     self.fine_tune_loss(human_hl, human_pred) / self.global_hl_batch_size
        # )
        # mouse_loss = (
        #     self.fine_tune_loss(mouse_hl, mouse_pred) / self.global_hl_batch_size
        # )

        self.human_loss_tracker.update_state(human_loss)
        self.human_pearson.update_state(human_hl, human_pred)

        self.mouse_loss_tracker.update_state(mouse_loss)
        self.mouse_pearson.update_state(mouse_hl, mouse_pred)

        # Only the probe metrics are logged at test time
        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        return {
            "resnet": self.resnet,
            "dropout_prob": self.dropout_prob,
            "global_hl_batch_size": self.global_hl_batch_size,
            "contrastive_checkpoint_epoch": self.contrastive_checkpoint_epoch,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def init_wandb_hl_fine_tune(args):
    project = "hl_fine_tune"

    if args["contrastive_run_dir"]:
        args["from_contrastive"] = True
    else:
        args["from_contrastive"] = False

    run_id = (
        f"{args['note']}-contrast_{args['from_contrastive']}-d_{args['dropout_prob']}-"
        f"seed_{args['rand_seed']}-bs_{args['global_hl_batch_size']}-"
        f"resnet_{args['resnet']}-lr_{args['lr']}-e_{args['number_epochs']}"
    )

    id_no_seed = (
        f"{args['note']}-contrast_{args['from_contrastive']}-d_{args['dropout_prob']}-"
        f"bs_{args['global_hl_batch_size']}-"
        f"resnet_{args['resnet']}-lr_{args['lr']}-e_{args['number_epochs']}"
    )

    # define and make path
    wandb_id_file_path_str = f"{args['wandb_run_dir']}/{project}-{run_id}"
    if not os.path.isdir(wandb_id_file_path_str):
        os.makedirs(wandb_id_file_path_str)

    args["id_no_seed"] = id_no_seed

    FLAGS.append_flags_into_file(f"{wandb_id_file_path_str}/config.json")

    run = wandb.init(
        id=run_id,
        project=project,
        config=args,
        dir=wandb_id_file_path_str,
    )
    args = wandb.config
    return run, args


def get_fine_tune_model(args):

    model = FineTuneModel(
        resnet=args["resnet"],
        dropout_prob=args["dropout_prob"],
        global_hl_batch_size=args["global_hl_batch_size"],
        num_classes=args["num_classes"],
        contrastive_checkpoint_epoch=args["contrastive_checkpoint_epoch"],
    )

    # if contrastive model
    if args["contrastive_run_dir"]:
        # which args should I be using here
        # Should probably load them
        checkpoint_dir = f"{args['contrastive_run_dir']}/checkpoints"
        assert os.path.isdir(checkpoint_dir)
        contrast_model, contrast_epoch = make_or_restore_model(
            checkpoint_dir,
            {"lr": 1, "hl_lr": 1, "mixed_precision": args["mixed_precision"]},
            checkpoint_epoch=args["contrastive_checkpoint_epoch"],
        )
        assert contrast_epoch != 0
        # set conv body to the contrast model conv body
        model.load_weights_from_contrastive_model(contrast_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args["lr"])

    if FLAGS.mixed_precision:
        policy = mixed_precision.Policy("mixed_float16")
        # Already set
        # mixed_precision.set_global_policy(policy)
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    model.compile(
        optimizer=optimizer,
    )
    return model


def main(argv):
    args = FLAGS.flag_values_dict()

    if FLAGS.mixed_precision:
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)

    # init wandb and load model
    run, args = init_wandb_hl_fine_tune(args)
    tf.random.set_seed(args["rand_seed"])

    # load datasets
    hl_train_human = load_hl_data(
        dataset_dir=args["hl_dataset_parent_dir"] + "/data0",
        batch_size=args["global_hl_batch_size"],
        partition="train",
    )
    hl_eval_human = load_hl_data(
        dataset_dir=args["hl_dataset_parent_dir"] + "/data0",
        batch_size=args["global_hl_batch_size"],
        partition="valid",
        mode="eval",
    )

    hl_train_mouse = load_hl_data(
        dataset_dir=args["hl_dataset_parent_dir"] + "/data1",
        batch_size=args["global_hl_batch_size"],
        partition="train",
    )
    hl_eval_mouse = load_hl_data(
        dataset_dir=args["hl_dataset_parent_dir"] + "/data1",
        batch_size=args["global_hl_batch_size"],
        partition="valid",
        mode="eval",
    )

    train_data = tf.data.Dataset.zip((hl_train_human.dataset, hl_train_mouse.dataset))
    eval_data = tf.data.Dataset.zip((hl_eval_human.dataset, hl_eval_mouse.dataset))

    n_training_steps = min(
        hl_train_human.batches_per_epoch(), hl_train_mouse.batches_per_epoch()
    )
    # load model
    model = get_fine_tune_model(args)
    model.build(input_shape=(None, 12288, 4))
    model.model.build(input_shape=(None, 12288, 4))

    pretraining_history = model.fit(
        train_data,
        epochs=args["number_epochs"],
        validation_data=eval_data,
        initial_epoch=0,
        steps_per_epoch=n_training_steps,
        callbacks=[
            # tf.keras.callbacks.ModelCheckpoint(
            #     filepath=run_dir+ "/checkpoints" + "/ckpt-{epoch}",
            #     mode='min',
            #     monitor='contrastive_loss',
            #     # save every 20 batches
            #     save_freq=50,
            #     save_best_only=True,
            #     verbose=1,
            # ),
            WandbCallback(
                monitor="mse_loss_half_life_human",
                mode="min",
                log_batch_frequency=50,
                save_model=False,
            )
        ],
    )


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_string("note", "", "note for wandb")
    flags.DEFINE_bool("mixed_precision", False, "Whether to use mixed precision")
    flags.DEFINE_float("dropout_prob", 0.1, "dropout probability")
    flags.DEFINE_integer("rand_seed", 3, "random seed")
    flags.DEFINE_integer(
        "num_classes", 2, "number of classes and so outputs in the model"
    )
    flags.DEFINE_integer("global_hl_batch_size", 64, "Total half life batch size")
    flags.DEFINE_string("resnet", "resnet_18", "Type of model to use")
    flags.DEFINE_float("lr", 3e-4, "hl fine tune learning rate")
    flags.DEFINE_integer("number_epochs", 100, "Number of epochs")
    flags.DEFINE_string(
        "wandb_run_dir",
        "/scratch/hdd001/home/phil/rna_contrast/runs",
        "Wandb run directory",
    )
    flags.DEFINE_string(
        "contrastive_run_dir",
        "",
        "Directory from which to load the model. If missing initializes new one",
    )
    flags.DEFINE_integer(
        "contrastive_checkpoint_epoch",
        0,
        "If set to 0 loads the latest epoch. Else provide the epoch from which to load",
    )

    flags.DEFINE_string(
        "hl_dataset_parent_dir",
        (
            "/scratch/hdd001/home/phil/rna_contrast"
            "/datasets/deeplearning/train_wosc/f0_c0"
        ),
        "Directory of human half life dataset",
    )

    app.run(main)
    # Contrastive pretraining
