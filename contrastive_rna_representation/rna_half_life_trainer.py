import time
import numpy as np
import tensorflow as tf
import wandb
from absl import app
from absl import flags
import sys
import os
import tensorflow_addons as tfa
from tensorflow.keras import mixed_precision

from contrastive_rna_representation.util import PearsonR, make_timestamp
from contrastive_rna_representation.saluki_dataset import RnaDataset
from contrastive_rna_representation.saluki_layers import SalukiModel
from contrastive_rna_representation.contrastive_model import make_or_restore_model
from contrastive_rna_representation.resnet import (
    dilated_small,
    dilated_medium,
    dilated_tiny,
    dilated_extra_tiny,
    dilated_small_constant_filters,
    not_dilated_small,
    dilated_small2,
    dilated_large,
    less_dilated_small,
    less_dilated_small2,
    less_dilated_medium,
    less_dilated_medium2,
)
from contrastive_rna_representation.convnext import (
    convnext_small,
    convnext_small2,
    convnext_medium,
    convnext_small2_deep
)
from contrastive_rna_representation.vit import VisionTransformer
from contrastive_rna_representation.vit2 import ViT


class Trainer:
    def __init__(
        self,
        args,
        train_data,
        eval_data,
        out_dir,
        test_data=None
    ):
        self.train_data = train_data
        self.eval_data = eval_data
        self.test_data = test_data

        self.batch_size = args["global_hl_batch_size"]
        self.out_dir = args["wandb_run_dir"]
        self.compiled = False

        # early stopping
        if args["fraction_of_train"] != 1:
            # min is 20 max is 120
            patience_denom = np.min([args["fraction_of_train"] * 7, 1])
            patience = 20 / patience_denom
            self.patience = np.min([patience, 120])
        elif args["n_samples"]:
            self.patience = 1000
        else:
            self.patience = 20
        print(
            f"batches per epoch: {[td.batches_per_epoch() for td in self.train_data]}"
        )
        # compute batches/epoch
        self.train_epoch_batches = [td.batches_per_epoch() for td in self.train_data]
        self.eval_epoch_batches = [ed.batches_per_epoch() for ed in self.eval_data]
        self.train_epochs_min = 1
        suggested_number_epochs = int(args["number_epochs"] / args["fraction_of_train"])
        max_number_epochs = 2000
        self.train_epochs_max = np.min([suggested_number_epochs, max_number_epochs])
        print(
            f"Number training epochs: {self.train_epochs_max},"
            f"Data fraction: {args['fraction_of_train']}"
        )

        # dataset
        self.num_datasets = len(self.train_data)
        self.num_val_datasets = len(self.eval_data)
        self.dataset_indexes = []
        for di in range(self.num_datasets):
            self.dataset_indexes += [di] * self.train_epoch_batches[di]
        self.dataset_indexes = np.array(self.dataset_indexes)

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
        # model.build(input_shape=(None, 12288, args["n_tracks"]))
        # model.compute_output_shape(input_shape=(None, 12288, args["n_tracks"]))
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

        elif args["resnet"] == "dilated_small2":
            resnet_model = dilated_small2

        elif args["resnet"] == "less_dilated_small":
            resnet_model = less_dilated_small

        elif args["resnet"] == "less_dilated_small2":
            resnet_model = less_dilated_small2

        elif args["resnet"] == "dilated_medium":
            resnet_model = dilated_medium

        elif args["resnet"] == "less_dilated_medium":
            resnet_model = less_dilated_medium

        elif args["resnet"] == "less_dilated_medium2":
            resnet_model = less_dilated_medium2

        elif args["resnet"] == "dilated_tiny":
            resnet_model = dilated_tiny

        elif args["resnet"] == "dilated_extra_tiny":
            resnet_model = dilated_extra_tiny

        elif args["resnet"] == "dilated_small_constant_filters":
            resnet_model = dilated_small_constant_filters

        elif args["resnet"] == "not_dilated_small":
            resnet_model = not_dilated_small

        elif args["resnet"] == "convnext_small":
            resnet_model = convnext_small

        elif args["resnet"] == "convnext_small2":
            resnet_model = convnext_small2

        elif args["resnet"] == "convnext_small2_dreep":
            resnet_model = convnext_small2_deep

        elif args["resnet"] == "convnext_medium":
            resnet_model = convnext_medium

        elif args["resnet"] == "dilated_large":
            resnet_model = dilated_large

        elif args["resnet"] == "saluki":
            model = SalukiModel(l2_scale=args["l2_scale_weight_decay"])

        elif args["resnet"] == "vit":
            model = VisionTransformer(
                patch_size=128,
                hidden_size=256,
                depth=6,
                num_heads=6,
                mlp_dim=256,
                num_classes=args["num_classes"],
                sd_survival_probability=0.9,
                attention_bias=True,
            )

        elif args["resnet"] == "vit2":
            model = ViT(
                seq_len=12288,
                patch_size=64,
                num_classes=args['num_classes'],
                dim=184,
                depth=6,
                heads=4,
                mlp_dim=184,
                dropout=0.3,
                emb_dropout=0.3
            )

        if resnet_model:
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

            # for loading old models
            #checkpoint_dir = f"{args['contrastive_run_dir']}/checkpoints/ckpt-{args['contrastive_checkpoint_epoch']}"
            #assert os.path.isdir(checkpoint_dir)
            #contrast_model = tf.keras.models.load_model(checkpoint_dir, compile=False,)
            #contrast_epoch = args['contrastive_checkpoint_epoch']

            
            checkpoint_dir = f"{args['contrastive_run_dir']}/checkpoints"
            assert os.path.isdir(checkpoint_dir)
            contrast_model, contrast_epoch = make_or_restore_model(
                checkpoint_dir,
                {
                    "lr": 1,
                    "hl_lr": 1,
                    "mixed_precision": args['mixed_precision'],
                    "weight_decay": args["weight_decay"],
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
            self.load_weights_from_contrastive_model(
                model, contrast_model, args["n_tracks"], args["load_pool_weights"]
            )
        if args['resnet'] != 'saluki' and args['resnet'] != 'vit' and args['resnet'] != 'vit2':
            model.body.trainable = args['train_full_model']

        if args["second_optim_lr"]:
            self.make_multi_optimizer(args, model)

        return model

    def fit2(self, model, args):
        if not self.compiled:
            self.compile(model, args)

        ################################################################
        # prep

        # metrics
        train_mse_loss, train_loss, train_r = [], [], []
        valid_mse_loss, valid_loss, valid_r = [], [], []

        for di in range(max(self.num_val_datasets, self.num_datasets)):
            train_mse_loss.append(tf.keras.metrics.Mean(name="train%d_mse_loss" % di))
            train_loss.append(tf.keras.metrics.Mean(name="train%d_loss" % di))
            train_r.append(PearsonR(name="train%d_r" % di))
            valid_mse_loss.append(tf.keras.metrics.Mean(name="valid%d_mse_loss" % di))
            valid_loss.append(tf.keras.metrics.Mean(name="valid%d_loss" % di))
            valid_r.append(PearsonR(name="valid%d_r" % di))

        @tf.function
        def train_step0(x, y):
            if args["train_full_model"]:
                trainable_vars = model.trainable_variables
            if not args["train_full_model"]:
                trainable_vars = (
                    model.fc.trainable_variables + model.avgpool.trainable_variables
                )

            with tf.GradientTape() as tape:
                pred = model.supervised_inference(x, training=True)[:, 0:1]
                mse_loss = self.loss_fn(y, pred)
                loss = mse_loss + sum(model.losses)
                if args['mixed_precision']:
                    scaled_loss = self.optimizer.get_scaled_loss(loss)

            # Get unscaled gradients
            if args['mixed_precision']:
                scaled_gradients = tape.gradient(scaled_loss, trainable_vars)
                gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
            else:
                gradients = tape.gradient(loss, trainable_vars)

            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            train_loss[0](loss)
            train_mse_loss[0](mse_loss)
            train_r[0](y, pred)

        @tf.function
        def eval_step0(x, y):
            pred = model.supervised_inference(x, training=False)[:, 0:1]
            mse_loss = self.loss_fn(y, pred)
            loss = mse_loss + sum(model.losses)
            valid_mse_loss[0](mse_loss)
            valid_loss[0](loss)
            valid_r[0](y, pred)

        @tf.function
        def train_step1(x, y):
            if args["train_full_model"]:
                trainable_vars = model.trainable_variables
            if not args["train_full_model"]:
                trainable_vars = (
                    model.fc.trainable_variables + model.avgpool.trainable_variables
                )

            with tf.GradientTape() as tape:
                pred = model.supervised_inference(x, training=True)[:, 1:2]
                loss = self.loss_fn(y, pred) + sum(model.losses)
                if args['mixed_precision']:
                    scaled_loss = self.optimizer.get_scaled_loss(loss)

            # Get unscaled gradients
            if args['mixed_precision']:
                scaled_gradients = tape.gradient(scaled_loss, trainable_vars)
                gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
            else:
                gradients = tape.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            train_loss[1](loss)
            train_r[1](y, pred)

        @tf.function
        def eval_step1(x, y):
            pred = model.supervised_inference(x, training=False)[:, 1:2]
            mse_loss = self.loss_fn(y, pred)
            loss = mse_loss + sum(model.losses)
            valid_loss[1](loss)
            valid_mse_loss[1](mse_loss)
            valid_r[1](y, pred)

        def update_checkpoint_values(ckpt, val_best, unimproved):
            ckpt.listed = [tf.Variable(val_best)]
            ckpt.listed.append(tf.Variable(unimproved))

            ckpt.mapped = {'val_best': ckpt.listed[0]}
            ckpt.mapped['unimproved'] = ckpt.listed[1]
            return ckpt


        def train_epoch(datasets, dataset_indexes, ei):
            perf = dict()
            # shuffle datasets
            n_batches = int(len(dataset_indexes))
            np.random.shuffle(dataset_indexes)

            # get iterators
            train_data_iters = [iter(td.dataset) for td in datasets]
            # train
            t0 = time.time()

            # limit the number of batches seen during training
            # time if fraction of data is set
            for di in dataset_indexes[:n_batches]:
                x, y = safe_next(train_data_iters[di])
                if di == 0:
                    train_step0(x, y)
                else:
                    train_step1(x, y)
            print(f"Epoch {ei} - {(time.time() - t0)}s - {make_timestamp()}")

            for di in range(len(datasets)):
                perf["epoch"] = ei
                perf["mse_loss_half_life_human"] = (train_loss[di].result().numpy())
                perf["human_pearson"] = (train_r[di].result().numpy())
                perf["mse_only_loss_half_life_human"]=train_mse_loss[di].result().numpy()

            # reset metrics
            return perf

        def eval_epoch(
            datasets,
            unimproved,
            valid_best,
            manager,
            ckpt,
            save_model=False
        ):
            # print training accuracy
            perf = dict()
            for di in range(len(datasets)):
                # evaluate
                for x, y in datasets[di].dataset:
                    if di == 0:
                        eval_step0(x, y)
                    else:
                        eval_step1(x, y)

                # print validation accuracy
                print("  Data %d" % di, end="")
                print(" - train_loss: %.4f" % train_loss[di].result().numpy(), end="")
                print(" - train_r: %.4f" % train_r[di].result().numpy(), end="")
                print(" - valid_loss: %.4f" % valid_loss[di].result().numpy(), end="")
                print(" - valid_r: %.4f" % valid_r[di].result().numpy(), end='')

                early_stop_stat = valid_loss[di].result().numpy()
                # check best
                if early_stop_stat + 1e-4 < valid_best[di] :
                    print(" - best!", end="")
                    unimproved[di] = 0
                    valid_best[di] = early_stop_stat
                    ckpt = update_checkpoint_values(ckpt, val_best, unimproved)

                    if di == 0:
                        # only save model when best human
                        manager.save()
                        if save_model:
                            model.save(
                                "%s/model%d_check" % (self.out_dir, 0),
                                include_optimizer=False,
                                save_format="tf",
                            )
                else:
                    unimproved[di] += 1

                if di == 0:
                    perf["val_mse_loss_half_life_human"] = valid_loss[di].result().numpy()
                    perf["val_human_pearson"] = (valid_r[di].result().numpy())
                    perf["val_mse_only_loss_half_life_human"] = valid_mse_loss[di].result().numpy()

                if di == 1:
                    perf["val_mse_loss_half_life_mouse"]=valid_loss[di].result().numpy()
                    perf["val_mouse_pearson"] = (valid_r[di].result().numpy())
                    perf["val_mse_only_loss_half_life_mouse"] = valid_mse_loss[di].result().numpy()

                train_loss[di].reset_states()
                train_r[di].reset_states()
                train_mse_loss[di].reset_states()

                valid_loss[di].reset_states()
                valid_r[di].reset_states()
                valid_mse_loss[di].reset_states()

                print("", flush=True)
            return perf, unimproved, valid_best


        # improvement variables
        val_best = [np.inf] * self.num_val_datasets
        unimproved = [0] * self.num_val_datasets

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
            if ei >= self.train_epochs_min and np.min(unimproved) > self.patience:
                break
            else:
                performance_dict = dict()
                train_perf = train_epoch(self.train_data, self.dataset_indexes, ei)
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
        negative_loss = [-1 for _ in range(len(self.eval_data))]

        print('Final eval:')
        eval_perf, _, _ = eval_epoch(
            self.eval_data, unimproved, negative_loss, manager, ckpt
        )

        top_perf_val = dict()
        top_perf_val["max_val_human_pearson"] = eval_perf["val_human_pearson"]
        top_perf_val["min_val_human_loss"] = eval_perf["val_mse_only_loss_half_life_human"]
        top_perf_val["max_val_mouse_pearson"] = eval_perf["val_mouse_pearson"]
        top_perf_val["min_val_mouse_loss"] = eval_perf["val_mse_only_loss_half_life_mouse"]

        if args['only_human']:
            # evaluate it using the human head
            eval_perf, _, _ = eval_epoch(
                (self.eval_data[1],), unimproved, negative_loss, manager, ckpt
            )
            top_perf_val["max_val_mouse_pearson"] = eval_perf["val_human_pearson"]
            top_perf_val["min_val_mouse_loss"] = eval_perf["val_mse_only_loss_half_life_human"]

        wandb.log(top_perf_val)

        test_perf, _, _ = eval_epoch(
            self.test_data, unimproved, negative_loss, manager, ckpt
        )

        top_perf_test = dict()
        top_perf_test["max_test_human_pearson"] = test_perf["val_human_pearson"]
        top_perf_test["min_test_human_loss"] = test_perf["val_mse_only_loss_half_life_human"]
        top_perf_test["max_test_mouse_pearson"] = test_perf["val_mouse_pearson"]
        top_perf_test["min_test_mouse_loss"] = test_perf["val_mse_only_loss_half_life_mouse"]

        if args['only_human']:
            # evaluate it using the human head
            test_perf, _, _ = eval_epoch(
                (self.test_data[1],), unimproved, negative_loss, manager, ckpt
            )
            top_perf_test["max_test_mouse_pearson"] = test_perf["val_human_pearson"]
            top_perf_test["min_test_mouse_loss"] = test_perf["val_mse_only_loss_half_life_human"]

        wandb.log(top_perf_test)

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
                decay_steps=20_000 / args["global_hl_batch_size"] * 2,
                decay_rate=args["exponential_decay_rate"],
            )
        elif args["lr_schedule"] == "cosine_restarts":
            self.lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate,
                first_decay_steps=20_000 / args["global_hl_batch_size"] * 15,
                alpha=0.0001,
            )
        elif args["lr_schedule"] == "cosine":
            self.lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate,
                decay_steps=20_000 / args["global_hl_batch_size"] * 50,
                alpha=0.0001,
                # warmup_steps=20_000 / args["global_hl_batch_size"] * 10,
            )
        else:
            self.lr_schedule = ""

        if args['optimizer'] == 'swa':
            self.optimizer = tfa.optimizers.SWA(
                optimizer="adam",
                lr=args["lr"],
                global_clipnorm=global_clipnorm,
            )
        elif args['optimizer'] == 'adamw':
            self.optimizer = tfa.optimizers.AdamW(
                learning_rate=args["lr"],
                weight_decay=args["weight_decay"],
                clipnorm=args['clipnorm']
            )

        else:
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=args["lr"],
                global_clipnorm=global_clipnorm,
                amsgrad=False,
                beta_1=0.90,
                beta_2=0.998,
            )
        if args["mixed_precision"]:
            # Already set
            self.optimizer = mixed_precision.LossScaleOptimizer(
                self.optimizer
            )


    def make_multi_optimizer(self, args, model):
        # if we need a second optimizer
        assert model
        if args["lr_schedule"] == "exponential":
            # every 2 epochs reduce by this `exponential_decay_rate`
            lr_schedule_1 = tf.keras.optimizers.schedules.ExponentialDecay(
                args["lr"],
                decay_steps=20_000 / args["global_hl_batch_size"] * 2,
                decay_rate=args["exponential_decay_rate"],
            )
            lr_schedule_2 = tf.keras.optimizers.schedules.ExponentialDecay(
                args["second_optim_lr"],
                decay_steps=20_000 / args["global_hl_batch_size"] * 2,
                decay_rate=args["exponential_decay_rate"],
            )

        elif args["lr_schedule"] == "cosine_restarts":
            lr_schedule_1 = tf.keras.optimizers.schedules.CosineDecayRestarts(
                args["lr"],
                first_decay_steps=20_000 / args["global_hl_batch_size"] * 15,
                alpha=0.0001,
            )
            lr_schedule_2 = tf.keras.optimizers.schedules.CosineDecayRestarts(
                args["second_optim_lr"],
                first_decay_steps=20_000 / args["global_hl_batch_size"] * 15,
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


def init_wandb_hl_fine_tune(args, project="hl_fine_tune_4"):
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


def safe_next(data_iter, retry=5, sleep=10):
    attempts = 0
    d = None
    while d is None and attempts < retry:
        try:
            d = next(data_iter)
        except tf.python.framework.errors_impl.AbortedError:
            print(
                "AbortedError, which has previously indicated NFS daemon restart.",
                file=sys.stderr,
            )
            time.sleep(sleep)
        attempts += 1

    if d is None:
        # let it crash
        d = next(data_iter)

    return d


def main(argv):
    args = FLAGS.flag_values_dict()
    run, args, out_dir = init_wandb_hl_fine_tune(args, project="hl_fine_tune_4")
    print(args)

    if args["mixed_precision"]:
        print(f"Activating mixed precision {args['mixed_precision']}")
        mixed_precision.set_global_policy('mixed_float16')

    # read datasets
    train_data = []
    eval_data = []
    test_data = []
    args["hl_dataset_parent_dir"]
    tf.random.set_seed(args["rand_seed"])

    if args['n_tracks'] == 4:
        codons=False
        splice=False
    else:
        codons=True
        splice=True

    # load_datasets
    for data_dir in [
        "/data0",
        "/data1",
    ]:
        data_dir = f"{args['hl_dataset_parent_dir']}{data_dir}"
        # load train data
        if args["only_human"] and data_dir.split("/")[-1] == "data1":
            # if train only on human and data1 == mouse SKIP LOADING
            pass
        else:
            # Else load data
            train_data.append(
                RnaDataset(
                    data_dir,
                    split_label="train",
                    batch_size=args["global_hl_batch_size"],
                    shuffle_buffer=1024,
                    mode="train",
                    codons=codons,
                    splice=splice,
                    fraction_of_train=args["fraction_of_train"],
                    n_samples=args["n_samples"],
                    add_positional_encoding=args["add_positional_encoding"],
                )
            )

        # load eval data
        eval_data.append(
            RnaDataset(
                data_dir,
                split_label="valid",
                batch_size=args["global_hl_batch_size"],
                mode="eval",
                codons=codons,
                splice=splice,
                add_positional_encoding=args["add_positional_encoding"],
            )
        )
        # load eval data
        test_data.append(
            RnaDataset(
                data_dir,
                split_label="test",
                batch_size=args["global_hl_batch_size"],
                mode="eval",
                codons=codons,
                splice=splice,
                add_positional_encoding=args["add_positional_encoding"],
            )
        )

    # initialize trainer
    seqnn_trainer = Trainer(
        args,
        train_data,
        eval_data,
        out_dir,
        test_data=test_data,
    )
    model = seqnn_trainer.load_model(args)
    # compile model
    seqnn_trainer.compile(model, args)

    # fit
    seqnn_trainer.fit2(model, args)


if __name__ == "__main__":
    FLAGS = flags.FLAGS

    flags.DEFINE_bool("mixed_precision", False, "Whether to use mixed precision")
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
    flags.DEFINE_string("optimizer", 'adam', "optimizer to use")
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
    flags.DEFINE_integer("kernel_size", 3, "size of kernel to use for convolutions")
    flags.DEFINE_float("l2_scale_weight_decay", 0, "l2 regularization for weights")
    flags.DEFINE_string("pooling_layer", "avgpool", "What kind of layer to use to pool")
    flags.DEFINE_string("norm_type", "batchnorm", "Type of normalization to use")
    flags.DEFINE_integer(
        "num_classes", 2, "number of classes and so outputs in the model"
    )
    flags.DEFINE_float("dropout_prob", 0.1, "dropout probability")
    flags.DEFINE_string("fc_head", "", "Type of fully connected head to use")
    flags.DEFINE_string("max_pool", 'max_pool', "What kind of pooling layer to use")
    flags.DEFINE_bool(
        "save_model",
        False,
        "Whether to save model during training",
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
    flags.DEFINE_bool("only_human", False, "train only on human data")
    flags.DEFINE_integer(
        "n_tracks", 6, "number of data tracks that the model is trained on"
    )
    flags.DEFINE_string(
        "wandb_run_dir",
        "/scratch/hdd001/home/phil/rna_contrast/runs",
        "Wandb run directory",
    )
    flags.DEFINE_string(
        "kernel_initializer", "he_normal", "What kind of initialization to use"
    )

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

    # data stuff
    flags.DEFINE_bool(
        "add_positional_encoding",
        False,
        "Whether to add positional encoding to the input sequence",
    )
    flags.DEFINE_string(
        "hl_dataset_parent_dir",
        "/scratch/hdd001/home/phil/rna_contrast/datasets/deeplearning/train_wosc/f0_c0",
        "Directory of human half life dataset",
    )

    app.run(main)
    # Contrastive pretraining
