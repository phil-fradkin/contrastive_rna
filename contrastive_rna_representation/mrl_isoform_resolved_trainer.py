import time
import numpy as np
import tensorflow as tf
import wandb
from absl import app
from absl import flags
import os
import tensorflow_addons as tfa

from contrastive_rna_representation.util import PearsonR, load_mrl_resolved_tf_data, make_timestamp
from contrastive_rna_representation.saluki_layers import SalukiModelFunctional, SalukiModel
from contrastive_rna_representation.contrastive_model import make_or_restore_model
from contrastive_rna_representation.resnet import (
    dilated_small,
    dilated_medium,
    dilated_tiny,
    dilated_extra_tiny,
    dilated_small_constant_filters,
    not_dilated_small,
    dilated_small2,
    less_dilated_medium,
    less_dilated_small2,
)


class Trainer:
    def __init__(
        self,
        args,
        train_data,
        eval_data,
        test_data,
        n_train_batches,
        n_eval_batches,
        n_test_batches,
        out_dir,
    ):
        self.train_data = train_data
        self.eval_data = eval_data
        self.test_data = test_data

        self.batch_size = args["global_hl_batch_size"]
        self.out_dir = args["wandb_run_dir"]
        self.compiled = False
        self.train_epoch_batches = n_train_batches
        self.eval_epoch_batches = n_eval_batches
        self.test_epoch_batches = n_test_batches
        self.train_epochs_min = 1
        self.train_epochs_max = args["number_epochs"]

        # dataset
        self.num_datasets = 1

        # early stopping
        if args["fraction_of_train"] != 1:
            patience_denom = np.min([args["fraction_of_train"] * 30, 1])
            self.patience = 20 / patience_denom
        elif args["n_samples"]:
            self.patience = 1000
        else:
            self.patience = 20
        print(
            f"batches per epoch: {n_train_batches}"
        )
        # compute batches/epoch

        # loss
        self.loss = "mse"
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        # optimizer
        self.make_optimizer(args)

        # wandb
        FLAGS.append_flags_into_file(f"{out_dir}/config.json")

        self.out_dir = out_dir

    def compile(self, model, args):
        sample = np.random.randn(10, 12288, args["n_tracks"])
        model.supervised_inference(sample)
        model(sample)
        model.contrastive_inference(sample)
        model.projection_head = None

        model_metrics = [PearsonR()]

        model.compile(
            loss=self.loss_fn, optimizer=self.optimizer, metrics=model_metrics
        )
        model.build(input_shape=(None, 12288, args["n_tracks"]))
        model.compute_output_shape(input_shape=(None, 12288, args["n_tracks"]))
        print(model.summary())
        self.compiled = True

    def load_weights_from_contrastive_model(
        self, model, contrastive_model, n_tracks, load_pool_weights
    ):
        sample = np.random.randn(10, 12288, n_tracks)
        model.supervised_inference(sample)
        model(sample)
        model.body.set_weights(contrastive_model.model.body.get_weights())

        if load_pool_weights:
            model.avgpool.set_weights(contrastive_model.model.avgpool.get_weights())

    def load_model(self, args):
        resnet_model = None

        if args["resnet"] == "dilated_small":
            resnet_model = dilated_small

        if args["resnet"] == "dilated_small2":
            resnet_model = dilated_small2

        elif args["resnet"] == "dilated_medium":
            resnet_model = dilated_medium

        elif args["resnet"] == "less_dilated_medium":
            resnet_model = less_dilated_medium

        elif args["resnet"] == "less_dilated_small2":
            resnet_model = less_dilated_small2

        elif args["resnet"] == "dilated_tiny":
            resnet_model = dilated_tiny

        elif args["resnet"] == "dilated_extra_tiny":
            resnet_model = dilated_extra_tiny

        elif args["resnet"] == "dilated_small_constant_filters":
            resnet_model = dilated_small_constant_filters

        elif args["resnet"] == "not_dilated_small":
            resnet_model = not_dilated_small

        elif args["resnet"] == "saluki":
            model = SalukiModel(l2_scale=args["l2_scale_weight_decay"])


        if resnet_model:
            model = resnet_model(
                args["num_classes"],
                dropout_prob=args["dropout_prob"],
                norm_type=args["norm_type"],
                l2_scale=args["l2_scale_weight_decay"],
                kernel_initializer=args["kernel_initializer"],
                pooling_layer=args["pooling_layer"],
                block_type=args["block_type"],
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
                    "weight_decay": args["weight_decay"],
                    'l2_scale_weight_decay': 0,
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
            self.load_weights_from_contrastive_model(
                model, contrast_model, args["n_tracks"], args["load_pool_weights"]
            )

        if args["second_optim_lr"]:
            self.make_multi_optimizer(args, model)

        return model

    def fit(self, model, args):
        if not self.compiled:
            self.compile(model, args)

        ################################################################
        # prep

        # metrics
        train_loss = tf.keras.metrics.Mean(name="train_loss")
        train_r = PearsonR(name="train_r")
        valid_loss = tf.keras.metrics.Mean(name="valid_loss")
        valid_r = PearsonR(name="valid_r")
        valid_mse_loss = tf.keras.metrics.Mean(name="valid_loss")
        sanity_check = tf.keras.metrics.Mean(name="y_val")

        ################################################################
        # Training and Evaluation functions
        @tf.function
        def train_step(x, y):
            if args["train_full_model"]:
                trainable_vars = model.trainable_variables
            if not args["train_full_model"]:
                trainable_vars = (
                    model.fc.trainable_variables + model.avgpool.trainable_variables
                )

            with tf.GradientTape() as tape:
                pred = model.supervised_inference(x, training=True)[:, 0:1]
                loss = self.loss_fn(y, pred) + sum(model.losses)
            train_loss(loss)
            train_r(y, pred)
            sanity_check(y)
            gradients = tape.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        @tf.function
        def eval_step(x, y):
            pred = model.supervised_inference(x, training=False)[:, 0:1]
            mse_loss = self.loss_fn(y, pred)
            loss = mse_loss + sum(model.losses)
            valid_loss(loss)
            valid_r(y, pred)
            valid_mse_loss(mse_loss)

        def update_checkpoint_values(ckpt, val_best, unimproved):
            ckpt.listed = [tf.Variable(val_best)]
            ckpt.listed.append(tf.Variable(unimproved))

            ckpt.mapped = {'val_best': ckpt.listed[0]}
            ckpt.mapped['unimproved'] = ckpt.listed[1]
            return ckpt

        def train_epoch(dataset, ei):
            perf = dict()
            # train
            t0 = time.time()

            # limit the number of batches seen during training time if fraction of data is set
            for x, y in dataset:
                train_step(x, y)
            print(f"Epoch {ei} - {(time.time() - t0)}s - {make_timestamp()}")
            print(" - train_loss: %.4f" % train_loss.result().numpy(), end="")
            print(" - train_r: %.4f" % train_r.result().numpy(), end="")

            perf["epoch"] = ei
            perf["mse_loss_mrl"] = (train_loss.result().numpy())
            perf["pearson_mrl"] = (train_r.result().numpy())

            # reset metrics
            return perf

        def eval_epoch(
            dataset,
            unimproved,
            valid_best,
            manager,
            ckpt,
            save_model=False
        ):
            # print training accuracy
            perf = dict()
            for x, y in dataset:
                eval_step(x, y)

            # print validation accuracy
            print(" - valid_loss: %.4f" % valid_loss.result().numpy(), end="")
            print(" - valid_r: %.4f" % valid_r.result().numpy(), end='')

            early_stop_stat = valid_loss.result().numpy()
            # check best
            if early_stop_stat < valid_best:
                print(" - best!", end="")
                unimproved = 0
                valid_best = early_stop_stat
                ckpt = update_checkpoint_values(ckpt, val_best, unimproved)

                manager.save()
                if save_model:
                    model.save(
                        "%s/model%d_check" % (self.out_dir, 0),
                        include_optimizer=False,
                        save_format="tf",
                    )
            else:
                unimproved += 1

            perf["val_mse_loss_mrl"] = valid_loss.result().numpy()
            perf["val_pearson_mrl"] = valid_r.result().numpy()
            perf["val_mse_only_loss_mrl"] = valid_mse_loss.result().numpy()

            train_loss.reset_states()
            train_r.reset_states()

            valid_loss.reset_states()
            valid_r.reset_states()
            valid_mse_loss.reset_states()

            print("", flush=True)
            return perf, unimproved, valid_best

        # improvement variables
        val_best = np.inf
        unimproved = 0

        # checkpoint manager
        ckpt = tf.train.Checkpoint(model=model, optimizer=self.optimizer)
        ckpt = update_checkpoint_values(ckpt, val_best, unimproved)

        ckpt_dir = f"{self.out_dir}/model"
        manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
        if manager.latest_checkpoint:
            ckpt.restore(manager.latest_checkpoint)
            val_best = ckpt.mapped['val_best'].numpy()
            unimproved = ckpt.mapped['unimproved'].numpy()

            ckpt_end = 5 + manager.latest_checkpoint.find("ckpt-")
            epoch_start = int(manager.latest_checkpoint[ckpt_end:])
            opt_iters = self.optimizer.iterations

            print(
                f"Checkpoint restored at epoch {epoch_start}, optimizer iteration {opt_iters}, "
                f"Best validation losses {val_best}, unimproved for {unimproved} epochs."
            )
        else:
            print("No checkpoints found.")
            epoch_start = 0

        ################################################################
        # training loop

        for ei in range(epoch_start, self.train_epochs_max):
            if ei >= self.train_epochs_min and unimproved > self.patience:
                break
            else:
                performance_dict = dict()
                train_perf = train_epoch(self.train_data, ei)
                eval_perf, unimproved, val_best = eval_epoch(
                    self.eval_data, unimproved, val_best, manager, ckpt,
                    save_model=args['save_model']
                )

                performance_dict.update(train_perf)
                performance_dict.update(eval_perf)

                # Log the performance metrics
                if type(self.optimizer) != tfa.optimizers.MultiOptimizer:
                    performance_dict["lr"] = self.optimizer.learning_rate.numpy()

                wandb.log(performance_dict)

                # Doesn't work for multi optimizer
                if type(self.optimizer) != tfa.optimizers.MultiOptimizer:
                    self.update_learning_rate_using_learning_rate_schedule(
                        args, unimproved
                    )

        # at the end of training conduct an evaluation
        ckpt.restore(manager.latest_checkpoint)
        # We don't want the model to be saved
        negative_loss = -1

        print('Final eval:')
        eval_perf, _, _ = eval_epoch(self.eval_data, unimproved, negative_loss, manager, ckpt)

        top_perf = dict()
        top_perf["min_val_mse_only_loss_mrl"] = eval_perf["val_mse_only_loss_mrl"]
        top_perf["max_val_pearson_mrl"] = eval_perf["val_pearson_mrl"]

        wandb.log(top_perf)

        print('Final test:')
        negative_loss = -1
        eval_perf, _, _ = eval_epoch(self.test_data, unimproved, negative_loss, manager, ckpt)

        top_perf = dict()
        top_perf["min_test_mse_only_loss_mrl"] = eval_perf["val_mse_only_loss_mrl"]
        top_perf["max_test_pearson_mrl"] = eval_perf["val_pearson_mrl"]
        wandb.log(top_perf)


    def update_learning_rate_using_learning_rate_schedule(self, args, unimproved):
        """
        Update the learning rate of the optimizer using the learning rate schedule.
        """
        assert not bool(args["lr_decay"]) or not bool(
            args["lr_schedule"]
        ), "Only one of lr_decay and lr_schedule can be True"

        if (
            (np.min(unimproved) % 9 == 0)
            and args["lr_decay"]
            and (np.min(unimproved) != 0)
        ):
            lower_lr = self.optimizer.learning_rate.numpy() / 5
            new_lr = tf.cast(tf.constant(np.max([lower_lr, 1e-5])), tf.float32)
            print(
                f"\nreducing LR from {self.optimizer.learning_rate.numpy()} to"
                f" {new_lr}!"
            )

            self.optimizer.lr.assign(new_lr)

        if args["lr_schedule"]:
            self.optimizer.lr.assign(self.lr_schedule(self.optimizer.iterations))

    def make_optimizer(self, args, model=None):
        initial_learning_rate = args["lr"]
        global_clipnorm = args["clipnorm"]

        if args["lr_schedule"] == "exponential":
            # every 2 epochs reduce by this `exponential_decay_rate`
            self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=10_000 / args["global_hl_batch_size"] * 2,
                decay_rate=args["exponential_decay_rate"],
            )
        elif args["lr_schedule"] == "cosine_restarts":
            self.lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate,
                first_decay_steps=10_000 / args["global_hl_batch_size"] * 15,
                alpha=0.0001,
            )
        else:
            self.lr_schedule = ""

        if args["stochastic_weight_averaging"]:
            self.optimizer = tfa.optimizers.SWA(
                optimizer="adam",
                lr=args["lr"],
                global_clipnorm=global_clipnorm,
            )
        else:
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=args["lr"],
                global_clipnorm=global_clipnorm,
                amsgrad=False,
                beta_1=0.90,
                beta_2=0.998,
            )

    def make_multi_optimizer(self, args, model):
        # if we need a second optimizer
        assert model
        if args["lr_schedule"] == "exponential":
            # every 2 epochs reduce by this `exponential_decay_rate`
            lr_schedule_1 = tf.keras.optimizers.schedules.ExponentialDecay(
                args["lr"],
                decay_steps=10_000 / args["global_hl_batch_size"] * 2,
                decay_rate=args["exponential_decay_rate"],
            )
            lr_schedule_2 = tf.keras.optimizers.schedules.ExponentialDecay(
                args["second_optim_lr"],
                decay_steps=10_000 / args["global_hl_batch_size"] * 2,
                decay_rate=args["exponential_decay_rate"],
            )

        elif args["lr_schedule"] == "cosine_restarts":
            lr_schedule_1 = tf.keras.optimizers.schedules.CosineDecayRestarts(
                args["lr"],
                first_decay_steps=10_000 / args["global_hl_batch_size"] * 15,
                alpha=0.0001,
            )
            lr_schedule_2 = tf.keras.optimizers.schedules.CosineDecayRestarts(
                args["second_optim_lr"],
                first_decay_steps=10_000 / args["global_hl_batch_size"] * 15,
                alpha=0.0001,
            )

        else:
            lr_schedule_1 = args["lr"]
            lr_schedule_2 = args["second_optim_lr"]

        optimizer1 = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule_1,
            global_clipnorm=args["clipnorm"],
            amsgrad=False,
            beta_1=0.90,
            beta_2=0.998,
        )
        optimizer2 = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule_2,
            global_clipnorm=args["clipnorm"],
            amsgrad=False,
            beta_1=0.90,
            beta_2=0.998,
        )

        optimizers = [optimizer1, optimizer2]

        if type(model.avgpool) == tf.keras.Sequential:
            avgpool_layers = model.avgpool.layers
        else:
            avgpool_layers = [model.avgpool]

        optimizers_and_layers = [
            (optimizers[0], model.fc.layers + avgpool_layers),
            (optimizers[1], model.body.layers),
        ]
        self.optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)

    def generate_plot(self, model, data, head_num):
        """generate predictions from model and then create a scatterplot using seaborn
        and annotate the plot with pearson correlation and p-value
        """
        from contrastive_rna_representation.util import plot_scatterplot_and_correlation

        all_y = []
        all_pred = []
        for x, y in data:
            pred = model(x)[:, head_num : head_num + 1]
            all_pred.extend(pred.numpy().squeeze())
            all_y.extend(y.numpy().squeeze())

        out = plot_scatterplot_and_correlation(
            all_y, all_pred, "Correlation of predictions", "half_life", "predictions"
        )
        return out


# TODO import from other
def init_wandb_fine_tune(args, project="mrl_isoform_resolved_0"):
    if args["contrastive_run_dir"]:
        args["from_contrastive"] = True
    else:
        args["from_contrastive"] = False

    run_id = (
        f"{args['note']}-pre_{args['from_contrastive']}-"
        f"{args['pooling_layer']}-"
        f"{args['resnet']}-lr_{args['lr']}-l2wd_{args['l2_scale_weight_decay']}-"
        f"d_{args['dropout_prob']}-seed_{args['rand_seed']}-"
        f"bs_{args['global_hl_batch_size']}-frac_{args['fraction_of_train']}"
    )

    id_no_seed = (
        f"{args['note']}-pre_{args['from_contrastive']}-"
        f"{args['pooling_layer']}-"
        f"{args['resnet']}-lr_{args['lr']}-l2wd_{args['l2_scale_weight_decay']}-"
        f"d_{args['dropout_prob']}-bs_{args['global_hl_batch_size']}-"
        f"frac_{args['fraction_of_train']}"
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
    return run, args, wandb_id_file_path_str


def main(argv):
    args = FLAGS.flag_values_dict()
    run, args, out_dir = init_wandb_fine_tune(args, project="mrl_fine_tune_3")
    print(args)

    # read datasets
    tf.random.set_seed(args["rand_seed"])

    if args["input_length"] == 12288:
        pad = True
    elif args["input_length"] == 855:
        pad = False
    else:
        raise ValueError()

    # load_datasets
    mrl_datasets, mrl_steps_per_epochs = load_mrl_resolved_tf_data(
        mrl_labels_df_path=args['mrl_labels_df_path'],
        mrl_tss_df_path=args['mrl_tss_df_path'],
        batch_size=args["global_hl_batch_size"],
        fraction_of_train=args["fraction_of_train"],
        n_tracks=args["n_tracks"],
        seed=args['rand_seed'],
    )

    mrl_train, mrl_val, mrl_test = mrl_datasets
    steps_per_epoch, steps_per_eval, steps_per_test = mrl_steps_per_epochs

    # initialize trainer
    seqnn_trainer = Trainer(
        args,
        train_data=mrl_train,
        eval_data=mrl_val,
        test_data=mrl_test,
        n_train_batches=steps_per_epoch,
        n_eval_batches=steps_per_eval,
        n_test_batches=steps_per_test,
        out_dir=out_dir,
    )
    model = seqnn_trainer.load_model(args)
    # compile model
    seqnn_trainer.compile(model, args)

    # fit
    seqnn_trainer.fit(model, args)


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_string("note", "", "note for wandb")
    flags.DEFINE_integer("rand_seed", 3, "random seed")
    flags.DEFINE_integer("global_hl_batch_size", 64, "Total half life batch size")
    flags.DEFINE_string("resnet", "dilated_small", "Type of model to use")

    # optimizer
    flags.DEFINE_float("lr", 3e-4, "hl fine tune learning rate")
    flags.DEFINE_float("clipnorm", 5.0, "Clip the norm of the gradients beyond that")
    flags.DEFINE_integer("number_epochs", 100, "Number of epochs")
    flags.DEFINE_bool("lr_decay", False, "Whether to perform weight decay")
    flags.DEFINE_string("lr_schedule", "", "Which learning rate scheduler to use")
    flags.DEFINE_float("weight_decay", 0, "Decay weights by this value every epoch")
    flags.DEFINE_bool(
        "stochastic_weight_averaging", False, "Average weights over multiple epochs"
    )
    flags.DEFINE_float(
        "exponential_decay_rate", 0.96, "Decay lr by this value every epoch"
    )
    flags.DEFINE_bool(
        "train_full_model",
        True,
        "train all the weights or just the fully connected layer",
    )
    flags.DEFINE_float("second_optim_lr", 0, "hl fine tune learning rate for body")

    # Model
    flags.DEFINE_string(
        "block_type", "dilated", "Block to use for resnet construction. dilated or sk"
    )
    flags.DEFINE_integer("kernel_size", 3, "size of kernel to use for convolutions")
    flags.DEFINE_float("l2_scale_weight_decay", 0, "l2 regularization for weights")
    flags.DEFINE_string("pooling_layer", "avgpool", "What kind of layer to use to pool")
    flags.DEFINE_string("norm_type", "batchnorm", "Type of normalization to use")
    flags.DEFINE_integer(
        "num_classes", 1, "number of classes and so outputs in the model"
    )
    flags.DEFINE_float("dropout_prob", 0.1, "dropout probability")
    flags.DEFINE_string("fc_head", "", "Type of fully connected head to use")
    flags.DEFINE_string("max_pool", 'max_pool', "What kind of pooling layer to use")
    flags.DEFINE_bool(
        "save_model",
        False,
        "Whether to save model during training",
    )
    flags.DEFINE_string(
        "kernel_initializer", "he_normal", "What kind of initialization to use"
    )

    # Data
    flags.DEFINE_float(
        "fraction_of_train",
        1.0,
        "Fraction of training_data to subsample",
    )
    flags.DEFINE_integer(
        "n_samples",
        0,
        "number of samples to take",
    )
    flags.DEFINE_string(
        "mrl_dataset_parent_dir",
        "/scratch/hdd001/home/phil/rna_contrast/karollus_utr5/data_dict.pkl",
        "Directory of mean ribosome load dataset",
    )
    flags.DEFINE_integer(
        "n_tracks", 6, "number of data tracks that the model is trained on"
    )
    flags.DEFINE_string(
        "wandb_run_dir",
        "/scratch/hdd001/home/phil/rna_contrast/runs",
        "Wandb run directory",
    )
    flags.DEFINE_integer(
        "input_length",
        855,
        "Length of the input sequence",
    )

    # contrastive model stuff
    flags.DEFINE_string(
        "contrastive_run_dir",
        "",
        "Directory from which to load the model. If missing initializes new one",
    )
    flags.DEFINE_string(
        "mrl_tss_df_path",
        "../data/mrl_isoform_resolved/mrl_list_of_tss.csv",
        "Directory from which to load the data for MRL transcription start sites",
    )
    flags.DEFINE_string(
        "mrl_labels_df_path",
        "../data/mrl_isoform_resolved/mrl_per_tss.csv",
        "Directory from which to load the data for MRL actual value",
    )
    flags.DEFINE_bool(
        "load_pool_weights",
        True,
        "Whether to load contrastive model weights for pooling layer",
    )
    flags.DEFINE_integer(
        "contrastive_checkpoint_epoch",
        0,
        "If set to 0 loads the latest epoch. Else provide the epoch from which to load",
    )

    # data stuff
    flags.DEFINE_bool(
        "add_positional_encoding",
        False,
        "Whether to add positional encoding to the input sequence",
    )

    app.run(main)
    # Contrastive pretraining
