import sys
import tensorflow as tf


class SalukiModelFunctional(tf.keras.Model):
    def __init__(
        self,
        activation="relu",
        rnn_type="gru",
        seq_length=12288,
        augment_shift=3,
        num_targets=1,
        heads=2,
        filters=64,
        kernel_size=5,
        dropout=0.3,
        l2_scale=0.001,
        ln_epsilon=0.007,
        num_layers=6,
        bn_momentum=0.90,
        residual=False,
        initializer="he_normal",
        seq_depth=6,
        go_backwards=True,
    ):
        super(SalukiModelFunctional, self).__init__()

        self.activation = activation
        self.rnn_type = rnn_type
        self.seq_length = seq_length
        self.augment_shift = augment_shift
        self.num_targets = num_targets
        self.heads = heads
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.l2_scale = l2_scale
        self.ln_epsilon = ln_epsilon
        self.num_layers = num_layers
        self.bn_momentum = bn_momentum
        self.residual = residual
        self.initializer = initializer
        self.seq_depth = seq_depth
        self.go_backwards = go_backwards

        self.build_model()

    def build_model(self):
        ###################################################
        # inputs
        ###################################################
        sequence = tf.keras.Input(
            shape=(self.seq_length, self.seq_depth), name="sequence"
        )
        current = sequence

        # augmentation
        if self.augment_shift != 0:
            current = StochasticShift(self.augment_shift, symmetric=False)(current)

        ###################################################
        # initial
        ###################################################

        # RNA convolution
        current = tf.keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding="valid",
            kernel_initializer=self.initializer,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_scale),
        )(current)
        current = current
        if self.residual:
            initial = current
            current = tf.keras.layers.LayerNormalization(epsilon=self.ln_epsilon)(
                current
            )
            current = activate(current, self.activation)
            current = tf.keras.layers.Conv1D(
                filters=self.filters,
                kernel_size=1,
                padding="valid",
                kernel_initializer=self.initializer,
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_scale),
            )(current)
            current = tf.keras.layers.Dropout(self.dropout)(current)
            current = Scale()(current)
            current = tf.keras.layers.Add()([initial, current])

        # middle convolutions
        for mi in range(self.num_layers):
            current = tf.keras.layers.LayerNormalization(epsilon=self.ln_epsilon)(
                current
            )
            current = activate(current, self.activation)
            current = tf.keras.layers.Conv1D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                padding="valid",
                kernel_initializer=self.initializer,
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_scale),
            )(current)
            current = tf.keras.layers.Dropout(self.dropout)(current)
            if self.residual:
                initial = current
                current = tf.keras.layers.LayerNormalization(epsilon=self.ln_epsilon)(
                    current
                )
                current = activate(current, self.activation)
                current = tf.keras.layers.Conv1D(
                    filters=self.filters,
                    kernel_size=1,
                    padding="valid",
                    kernel_initializer=self.initializer,
                    kernel_regularizer=tf.keras.regularizers.l2(self.l2_scale),
                )(current)
                current = tf.keras.layers.Dropout(self.dropout)(current)
                current = Scale()(current)
                current = tf.keras.layers.Add()([initial, current])
            current = tf.keras.layers.MaxPooling1D()(current)

        # aggregate sequence
        current = tf.keras.layers.LayerNormalization(epsilon=self.ln_epsilon)(current)
        current = activate(current, self.activation)
        rnn_layer = tf.keras.layers.GRU
        if self.rnn_type == "lstm":
            rnn_layer = tf.keras.layers.LSTM
        current = rnn_layer(
            self.filters,
            go_backwards=self.go_backwards,
            kernel_initializer=self.initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_scale),
        )(current)

        # attention = tf.keras.layers.LayerNormalization(epsilon=self.ln_epsilon)(current)
        # attention = layers.activate(attention, self.activation)
        # attention = tf.keras.layers.Dense(units=self.filters,
        #                                   kernel_initializer=self.initializer,
        #                                   kernel_regularizer=tf.keras.regularizers.l2(self.l2_scale))(attention)
        # attention = tf.keras.layers.Softmax(axis=-2)(attention)
        # current *= attention
        # current = tf.keras.layers.GlobalAveragePooling1D()(current)

        # penultimate
        current = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)(current)
        current = activate(current, self.activation)
        current = tf.keras.layers.Dense(
            self.filters,
            kernel_initializer=self.initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_scale),
        )(current)
        current = tf.keras.layers.Dropout(self.dropout)(current)
        if self.residual:
            initial = current
            current = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)(
                current
            )
            current = activate(current, self.activation)
            current = tf.keras.layers.Dense(
                self.filters,
                kernel_initializer=self.initializer,
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_scale),
            )(current)
            current = tf.keras.layers.Dropout(self.dropout)(current)
            current = Scale()(current)
            current = tf.keras.layers.Add()([initial, current])

        # final representation
        current = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)(current)
        current = activate(current, self.activation)

        ###################################################
        # compile model(s)
        ###################################################
        # self.models = []
        # for hi in range(self.heads):
        #     prediction = tf.keras.layers.Dense(
        #         self.num_targets,
        #         kernel_initializer=self.initializer,
        #     )(current)
        #     self.models.append(tf.keras.Model(inputs=sequence, outputs=prediction))

        # self.model = self.models[0]

        prediction = tf.keras.layers.Dense(
            self.heads,
            kernel_initializer=self.initializer,
        )(current)
        self.model = tf.keras.Model(inputs=sequence, outputs=prediction)
        print(self.model.summary())

    def call(self, inputs, training=False, **kwargs):
        x = self.model(inputs, training=training, **kwargs)
        return x

    def supervised_inference(self, inputs, training=False, **kwargs):
        x = self.model(inputs, training=training, **kwargs)
        return x

    def contrastive_inference(self, inputs, training=False):
        return None


class StochasticShift(tf.keras.layers.Layer):
    """Stochastically shift a one hot encoded DNA sequence."""

    def __init__(self, shift_max=0, symmetric=True, pad="uniform"):
        super(StochasticShift, self).__init__()
        self.shift_max = shift_max
        self.symmetric = symmetric
        if self.symmetric:
            self.augment_shifts = tf.range(-self.shift_max, self.shift_max + 1)
        else:
            self.augment_shifts = tf.range(0, self.shift_max + 1)
        self.pad = pad

    def call(self, seq_1hot, training=None):
        if training:
            shift_i = tf.random.uniform(
                shape=[], minval=0, dtype=tf.int64, maxval=len(self.augment_shifts)
            )
            shift = tf.gather(self.augment_shifts, shift_i)
            sseq_1hot = tf.cond(
                tf.not_equal(shift, 0),
                lambda: shift_sequence(seq_1hot, shift),
                lambda: seq_1hot,
            )
            return sseq_1hot
        else:
            return seq_1hot

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {"shift_max": self.shift_max, "symmetric": self.symmetric, "pad": self.pad}
        )
        return config


def shift_sequence(seq, shift, pad_value=0):
    """Shift a sequence left or right by shift_amount.
    Args:
    seq: [batch_size, seq_length, seq_depth] sequence
    shift: signed shift value (tf.int32 or int)
    pad_value: value to fill the padding (primitive or scalar tf.Tensor)
    """
    if seq.shape.ndims != 3:
        raise ValueError("input sequence should be rank 3")
    input_shape = seq.shape

    pad = pad_value * tf.ones_like(seq[:, 0 : tf.abs(shift), :])

    def _shift_right(_seq):
        # shift is positive
        sliced_seq = _seq[:, :-shift:, :]
        return tf.concat([pad, sliced_seq], axis=1)

    def _shift_left(_seq):
        # shift is negative
        sliced_seq = _seq[:, -shift:, :]
        return tf.concat([sliced_seq, pad], axis=1)

    sseq = tf.cond(
        tf.greater(shift, 0), lambda: _shift_right(seq), lambda: _shift_left(seq)
    )
    sseq.set_shape(input_shape)

    return sseq

def activate(current, activation, verbose=False):
    if verbose:
        print("activate:", activation)
    if activation == "relu":
        current = tf.keras.layers.ReLU()(current)
    # elif activation == "polyrelu":
    #     current = PolyReLU()(current)
    elif activation == "gelu":
        current = tf.keras.layers.Activation("gelu")(current)
    elif activation == "sigmoid":
        current = tf.keras.layers.Activation("sigmoid")(current)
    elif activation == "tanh":
        current = tf.keras.layers.Activation("tanh")(current)
    elif activation == "exp":
        current = tf.keras.layers.Activation("exp")(current)
    elif activation == "softplus":
        current = tf.keras.layers.Activation("softplus")(current)
    else:
        print('Unrecognized activation "%s"' % activation, file=sys.stderr)
        exit(1)

    return current


class Scale(tf.keras.layers.Layer):
    def __init__(self, axis=-1, initializer="zeros"):
        super(Scale, self).__init__()
        if isinstance(axis, (list, tuple)):
            self.axis = axis[:]
        elif isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError(
                "Expected an int or a list/tuple of ints for the "
                "argument 'axis', but received: %r" % axis
            )
        self.initializer = tf.keras.initializers.get(initializer)

    def build(self, input_shape):
        # input_shape = tensor_shape.TensorShape(input_shape)
        if not input_shape.ndims:
            raise ValueError("Input has undefined rank.")
        ndims = len(input_shape)

        # Convert axis to list and resolve negatives
        if isinstance(self.axis, int):
            self.axis = [self.axis]
        elif isinstance(self.axis, tuple):
            self.axis = list(self.axis)
        for idx, x in enumerate(self.axis):
            if x < 0:
                self.axis[idx] = ndims + x

        # Validate axes
        for x in self.axis:
            if x < 0 or x >= ndims:
                raise ValueError("Invalid axis: %d" % x)
        if len(self.axis) != len(set(self.axis)):
            raise ValueError("Duplicate axis: {}".format(tuple(self.axis)))

        param_shape = [input_shape[dim] for dim in self.axis]

        self.scale = self.add_weight(
            name="scale",
            shape=param_shape,
            initializer=self.initializer,
            trainable=True,
        )

    def call(self, x):
        # return x * math_ops.cast(self.scale, x.dtype)
        return x * self.scale

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "axis": self.axis,
                "initializer": tf.keras.initializers.serialize(self.initializer),
            }
        )
        return config


class SalukiConv1D(tf.keras.Model):
    def __init__(
        self,
        filters,
        kernel_size,
        initializer,
        l2_scale,
        dropout,
        ln_epsilon,
        residual=False,
    ):
        super(SalukiConv1D, self).__init__()

        self.residual = residual

        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=ln_epsilon)
        self.act1 = tf.keras.layers.ReLU()
        self.conv1 = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="valid",
            kernel_initializer=initializer,
            kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
        )
        self.d1 = tf.keras.layers.Dropout(dropout)

        if self.residual:
            self.res_ln = tf.keras.layers.LayerNormalization(epsilon=ln_epsilon)
            self.res_act = tf.keras.layers.ReLU()
            self.res_conv = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=1,
                padding="valid",
                kernel_initializer=initializer,
                kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
            )
            self.res_d = tf.keras.layers.Dropout(dropout)
            self.res_scale = Scale()

        self.max1 = tf.keras.layers.MaxPooling1D()

    def call(self, inputs, **kwargs):
        x = self.ln1(inputs)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.d1(x)

        if self.residual:
            residual = x
            x = self.res_ln(x)
            x = self.rest_act(x)
            x = self.res_conv(x)
            x = self.res_d(x)
            x = self.res_scale(x)
            x = tf.keras.layers.add([residual, x])

        x = self.max1(x)

        return x


def make_saluki_layer(
    num_layers,
    filters,
    kernel_size,
    initializer,
    l2_scale,
    dropout,
    ln_epsilon,
    residual=False,
):
    res_block = tf.keras.Sequential()

    for mi in range(num_layers):
        res_block.add(
            SalukiConv1D(
                filters,
                kernel_size,
                initializer,
                l2_scale,
                dropout,
                ln_epsilon,
                residual=residual,
            )
        )

    return res_block


class SalukiModel(tf.keras.Model):
    def __init__(
        self,
        activation="relu",
        rnn_type="gru",
        seq_length=12288,
        augment_shift=3,
        num_targets=1,
        heads=2,
        filters=64,
        kernel_size=5,
        dropout=0.3,
        l2_scale=0.001,
        ln_epsilon=0.007,
        num_layers=6,
        bn_momentum=0.90,
        residual=False,
        initializer="he_normal",
        seq_depth=6,
        go_backwards=True,
        projection_head_size=256,
        projection_body=1024,
    ):
        super(SalukiModel, self).__init__()

        self.residual = residual
        self.projection_head_size = projection_head_size
        self.projection_body = projection_body

        conv0 = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="valid",
            kernel_initializer=initializer,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
        )

        assert not self.residual

        shift = StochasticShift(augment_shift, symmetric=False)

        mid = make_saluki_layer(
            num_layers,
            filters,
            kernel_size,
            initializer,
            l2_scale,
            dropout,
            ln_epsilon,
            residual=residual,
        )

        ln = tf.keras.layers.LayerNormalization(epsilon=ln_epsilon)
        act = tf.keras.layers.ReLU()

        rnn_layer = tf.keras.layers.GRU(
            filters,
            go_backwards=go_backwards,
            kernel_initializer=initializer,
            kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
            input_shape=(None, filters)
        )
        bn2 = tf.keras.layers.BatchNormalization(momentum=bn_momentum)
        act2 = tf.keras.layers.ReLU()
        dense2 = tf.keras.layers.Dense(
            filters,
            kernel_initializer=initializer,
            kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
        )
        d2 = tf.keras.layers.Dropout(dropout)

        bn3 = tf.keras.layers.BatchNormalization(momentum=bn_momentum)
        act3 = tf.keras.layers.ReLU()

        final_fc = tf.keras.layers.Dense(
            heads,
            kernel_initializer=initializer,
            dtype="float32"
        )

        self.body = tf.keras.Sequential(
            [shift, conv0, mid, ln, act, ],
            name="conv_body",
        )

        self.fc = tf.keras.Sequential([
            rnn_layer,
            bn2,
            act2,
            dense2,
            d2,
            bn3,
            act3,
            final_fc,
        ], name='fc')

        self.avgpool = tf.keras.layers.GlobalAveragePooling1D()
        self.projection_head = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(projection_body),
                tf.keras.layers.experimental.SyncBatchNormalization(momentum=.9, epsilon=0.01),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.Dense(projection_body),
                tf.keras.layers.experimental.SyncBatchNormalization(momentum=.9, epsilon=0.01),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.Dense(projection_head_size, dtype="float32", use_bias=False),
            ],
            name="projection_head",
        )

    def call(self, inputs, training=False):
        x = self.body(inputs, training=training)
        x = self.fc(x)
        return x

    def supervised_inference(self, inputs, training=False):
        x = self.body(inputs, training=training)
        output = self.fc(x)
        return output

    def contrastive_inference(self, inputs, training=False):
        x = self.body(inputs, training=training)
        x = self.avgpool(x)
        output = self.projection_head(x, training=training)
        return output
