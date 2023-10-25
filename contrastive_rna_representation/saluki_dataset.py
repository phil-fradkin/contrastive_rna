# Copyright 2017 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from __future__ import print_function
import glob
import json
import os
import sys
import math

from natsort import natsorted
import numpy as np
import tensorflow as tf

from contrastive_rna_representation.my_layers import get_position_encoding

# TFRecord constants
TFR_INPUT = "sequence"
TFR_OUTPUT = "target"


def file_to_records(filename):
    return tf.data.TFRecordDataset(filename, compression_type="ZLIB")


class RnaDataset:
    def __init__(
        self,
        data_dir,
        split_label,
        batch_size,
        mode="eval",
        shuffle_buffer=1024,
        splice=False,
        codons=False,
        fraction_of_train=1.0,
        n_samples=0,
        add_positional_encoding=False,
    ):
        """Initialize basic parameters; run make_dataset."""

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.mode = mode
        self.split_label = split_label
        self.splice = splice
        self.codons = codons
        self.fraction_of_train = fraction_of_train
        self.n_samples = n_samples
        self.add_positional_encoding = add_positional_encoding

        # read data parameters
        data_stats_file = "%s/statistics.json" % self.data_dir
        with open(data_stats_file) as data_stats_open:
            data_stats = json.load(data_stats_open)
        self.length_t = data_stats["length_t"]

        # self.seq_depth = data_stats.get('seq_depth',4)
        self.target_length = data_stats["target_length"]
        self.num_targets = data_stats["num_targets"]

        if self.split_label == "*":
            self.num_seqs = 0
            for dkey in data_stats:
                if dkey[-5:] == "_seqs":
                    self.num_seqs += data_stats[dkey]
        else:
            self.num_seqs = data_stats["%s_seqs" % self.split_label]

        self.make_dataset()

    def batches_per_epoch(self):
        return math.ceil(self.num_seqs / self.batch_size)

    def make_parser(self):  # , rna_mode
        def parse_proto(example_protos):
            """Parse TFRecord protobuf."""

            feature_spec = {
                "lengths": tf.io.FixedLenFeature((1,), tf.int64),
                "sequence": tf.io.FixedLenFeature([], tf.string),
                "coding": tf.io.FixedLenFeature([], tf.string),
                "splice": tf.io.FixedLenFeature([], tf.string),
                "targets": tf.io.FixedLenFeature([], tf.string),
            }

            # parse example into features
            feature_tensors = tf.io.parse_single_example(
                example_protos, features=feature_spec
            )

            # decode targets
            targets = tf.io.decode_raw(feature_tensors["targets"], tf.float16)
            targets = tf.cast(targets, tf.float32)

            # get length
            seq_lengths = feature_tensors["lengths"]

            # decode sequence
            sequence = tf.io.decode_raw(feature_tensors["sequence"], tf.uint8)
            sequence = tf.one_hot(sequence, 4)
            sequence = tf.cast(sequence, tf.float32)

            # decode coding frame
            if self.codons:
                coding = tf.io.decode_raw(feature_tensors["coding"], tf.uint8)
                coding = tf.expand_dims(coding, axis=1)
                coding = tf.cast(coding, tf.float32)

            # decode splice
            if self.splice:
                splice = tf.io.decode_raw(feature_tensors["splice"], tf.uint8)
                splice = tf.expand_dims(splice, axis=1)
                splice = tf.cast(splice, tf.float32)

            # concatenate input tracks
            if self.codons and self.splice:
                inputs = tf.concat([sequence, coding, splice], axis=1)
            elif self.splice:
                inputs = tf.concat([sequence, splice], axis=1)
            elif self.codons:
                inputs = tf.concat([sequence, coding], axis=1)
            else:
                inputs = sequence

            # pad to zeros to full length
            paddings = [[0, self.length_t - seq_lengths[0]], [0, 0]]
            inputs = tf.pad(inputs, paddings)

            if self.add_positional_encoding:
                assert self.codons
                assert self.splice
                position_encoding = get_position_encoding(12288, 6, n=50_000)
                inputs = inputs + position_encoding

            return inputs, targets

        return parse_proto

    def make_dataset(self, cycle_length=4):
        """Make Dataset w/ transformations."""

        # collect tfrecords
        tfr_path = "%s/tfrecords/%s-*.tfr" % (self.data_dir, self.split_label)
        tfr_files = natsorted(glob.glob(tfr_path))

        # initialize tf.data
        if tfr_files:
            # dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)
            dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
        else:
            print("Cannot order TFRecords %s" % tfr_path, file=sys.stderr)
            dataset = tf.data.Dataset.list_files(tfr_path)

        # train
        if self.mode == "train":
            # repeat
            dataset = dataset.repeat()

            # interleave files
            dataset = dataset.interleave(
                map_func=file_to_records,
                cycle_length=cycle_length,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )

            # take fraction of train
            if self.fraction_of_train < 1.0:
                assert self.n_samples == 0
                self.num_seqs = int(self.num_seqs * self.fraction_of_train)
                dataset = dataset.take(self.num_seqs)

            if self.n_samples > 0:
                assert self.fraction_of_train == 1.0
                self.num_seqs = self.n_samples
                dataset = dataset.take(self.num_seqs)

            # shuffle
            dataset = dataset.shuffle(
                buffer_size=self.shuffle_buffer, reshuffle_each_iteration=True
            )

        else:
            # flat mix files
            dataset = dataset.flat_map(file_to_records)

        # map records to examples
        dataset = dataset.map(self.make_parser())  # self.rna_mode

        # batch
        dataset = dataset.batch(self.batch_size)

        # prefetch
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # hold on
        self.dataset = dataset

    def numpy(self, return_inputs=True, return_outputs=True):
        """Convert TFR inputs and/or outputs to numpy arrays."""
        with tf.name_scope("numpy"):
            # initialize dataset from TFRecords glob
            tfr_path = "%s/tfrecords/%s-*.tfr" % (self.data_dir, self.split_label)
            tfr_files = natsorted(glob.glob(tfr_path))
            if tfr_files:
                # dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)
                dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
            else:
                print("Cannot order TFRecords %s" % self.tfr_path, file=sys.stderr)
                dataset = tf.data.Dataset.list_files(self.tfr_path)

            # read TF Records
            dataset = dataset.flat_map(file_to_records)
            dataset = dataset.map(self.make_parser())
            dataset = dataset.batch(1)

        # initialize inputs and outputs
        seqs_1hot = []
        targets = []

        # collect inputs and outputs
        for seq_1hot, targets1 in dataset:
            # sequence
            if return_inputs:
                seqs_1hot.append(seq_1hot.numpy())

            # targets
            if return_outputs:
                targets.append(targets1.numpy())

        # make arrays
        seqs_1hot = np.array(seqs_1hot)
        targets = np.array(targets)

        # return
        if return_inputs and return_outputs:
            return seqs_1hot, targets
        elif return_inputs:
            return seqs_1hot
        else:
            return targets
