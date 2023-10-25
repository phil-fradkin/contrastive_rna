import tensorflow.keras.layers as kl
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
import numpy as np
import tensorflow_addons as tfa
import tensorflow.keras.backend as K


class DilatedConv1DBasic(tf.keras.layers.Layer):
    """Dillated convolutional layers

    - add_pointwise -> if True, add a 1by1 conv right after the first conv
    """

    def __init__(
        self,
        filter_num=21,
        kernel_size=8,
        padding="same",
        dropout_prob=0.1,
        dilation=2,
        max_pool='max_pool',
        norm_type="batchnorm",
        l2_scale=0,
        kernel_initializer="glorot_uniform",
    ):
        super(DilatedConv1DBasic, self).__init__()
        self.norm_type = norm_type
        self.filter_num = filter_num

        self.conv1 = kl.Conv1D(
            filters=filter_num,
            kernel_size=kernel_size,
            padding=padding,
            activation="relu",
            dilation_rate=dilation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
        )
        self.bn1 = return_norm_layer(norm_type)

        self.conv2 = kl.Conv1D(
            filters=filter_num,
            kernel_size=kernel_size,
            padding=padding,
            activation="relu",
            dilation_rate=dilation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
        )
        self.bn2 = return_norm_layer(norm_type)

        if dropout_prob:
            self.dropout = kl.Dropout(dropout_prob)
        else:
            self.dropout = IdentityLayer()

        if max_pool == 'max_pool':
            self.max_pool = kl.MaxPool1D(pool_size=2)
        elif max_pool == 'attn_pool':
            self.max_pool = SoftmaxPooling1D()
        else:
            self.max_pool = IdentityLayer()

    def call(self, inputs, training=False, **kwargs):
        """x = (None, 4)"""
        residual = self.downsample(inputs, training=training)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = tf.nn.relu(tf.keras.layers.add([residual, x]))
        output = self.max_pool(x)

        return output

    def build(self, input_shape):
        if input_shape[-1] != self.filter_num:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(kl.Conv1D(filters=self.filter_num, kernel_size=1))
            self.downsample.add(return_norm_layer(self.norm_type))
        else:
            self.downsample = IdentityLayer()


class DilatedConv1DBottleneck(tf.keras.layers.Layer):
    """Dillated convolutional layers

    - add_pointwise -> if True, add a 1by1 conv right after the first conv
    """

    def __init__(
        self,
        filter_num=21,
        kernel_size=8,
        padding="same",
        dropout_prob=0.1,
        dilation=2,
        max_pool='max_pool',
        norm_type="batchnorm",
        l2_scale=0,
        kernel_initializer="glorot_uniform",
    ):
        super(DilatedConv1DBottleneck, self).__init__()
        self.norm_type = norm_type
        self.filter_num = filter_num

        self.conv1 = kl.Conv1D(
            filters=filter_num,
            kernel_size=kernel_size,
            padding=padding,
            activation="relu",
            dilation_rate=dilation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
        )
        self.bn1 = return_norm_layer(norm_type)

        self.conv2 = kl.Conv1D(
            filters=filter_num,
            kernel_size=kernel_size,
            padding=padding,
            activation="relu",
            dilation_rate=dilation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
        )
        self.bn2 = return_norm_layer(norm_type)

        self.conv3 = kl.Conv1D(
            filters=filter_num * 4,
            kernel_size=1,
            padding=padding,
            activation="relu",
            dilation_rate=1,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
        )
        self.bn3 = return_norm_layer(norm_type)

        if dropout_prob:
            self.dropout = kl.Dropout(dropout_prob)
            # self.dropout = DropBlock1D(block_size=5, keep_prob=1 - dropout_prob)
        else:
            self.dropout = IdentityLayer()

        if max_pool == 'max_pool':
            self.max_pool = kl.MaxPool1D(pool_size=2)
        elif max_pool == 'attn_pool':
            self.max_pool = SoftmaxPooling1D()
        else:
            self.max_pool = IdentityLayer()

    def call(self, inputs, training=False, **kwargs):
        """x = (None, 4)"""
        residual = self.downsample(inputs, training=training)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.dropout(x, training=training)
        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x = tf.nn.relu(tf.keras.layers.add([residual, x]))
        output = self.max_pool(x)

        return output

    def build(self, input_shape):
        if input_shape[-1] != self.filter_num * 4:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(kl.Conv1D(filters=self.filter_num * 4, kernel_size=1))
            self.downsample.add(return_norm_layer(self.norm_type))
        else:
            self.downsample = IdentityLayer()


def make_dilated_basic_block_layer(
    filter_num=21,
    kernel_size=8,
    padding="same",
    dropout_prob=0.1,
    dilation=2,
    max_pool='max_pool',
    blocks=2,
    norm_type="batchnorm",
    kernel_initializer="glorot_uniform",
    l2_scale=0,
    increase_dilation=True,
    **kwargs,
):
    res_block = tf.keras.Sequential()

    res_block.add(
        DilatedConv1DBasic(
            filter_num=filter_num,
            kernel_size=kernel_size,
            padding=padding,
            dropout_prob=dropout_prob,
            dilation=dilation,
            max_pool=max_pool,
            norm_type=norm_type,
            kernel_initializer=kernel_initializer,
            l2_scale=l2_scale,
        )
    )

    for i in range(1, blocks):
        # have the option to increase dilation in
        # the subsequent layers or keep it constant
        if increase_dilation:
            new_dilation = dilation * (2**i)
        else:
            new_dilation = dilation
        res_block.add(
            DilatedConv1DBasic(
                filter_num=filter_num,
                kernel_size=kernel_size,
                padding=padding,
                dropout_prob=dropout_prob,
                dilation=new_dilation,
                max_pool='',
                norm_type=norm_type,
                kernel_initializer=kernel_initializer,
                l2_scale=l2_scale,
            )
        )

    return res_block


def make_dilated_bottleneck_block_layer(
    filter_num=21,
    kernel_size=8,
    padding="same",
    dropout_prob=0.1,
    dilation=2,
    max_pool='max_pool',
    blocks=2,
    norm_type="batchnorm",
    kernel_initializer="glorot_uniform",
    l2_scale=0,
    increase_dilation=True,
    **kwargs,
):
    res_block = tf.keras.Sequential()

    res_block.add(
        DilatedConv1DBottleneck(
            filter_num=filter_num,
            kernel_size=kernel_size,
            padding=padding,
            dropout_prob=dropout_prob,
            dilation=dilation,
            max_pool=max_pool,
            norm_type=norm_type,
            kernel_initializer=kernel_initializer,
            l2_scale=l2_scale,
        )
    )

    for i in range(1, blocks):
        # have the option to increase dilation in
        # the subsequent layers or keep it constant
        if increase_dilation:
            new_dilation = dilation * (2**i)
        else:
            new_dilation = dilation
        res_block.add(
            DilatedConv1DBottleneck(
                filter_num=filter_num,
                kernel_size=kernel_size,
                padding=padding,
                dropout_prob=dropout_prob,
                dilation=new_dilation,
                max_pool='',
                norm_type=norm_type,
                kernel_initializer=kernel_initializer,
                l2_scale=l2_scale,
            )
        )

    return res_block


def return_norm_layer(norm_type="batchnorm", **kwargs):
    if norm_type == "batchnorm":
        return tf.keras.layers.BatchNormalization( **kwargs)
    elif norm_type == "batchnorm_small_momentum":
        return tf.keras.layers.BatchNormalization(momentum=0.9, **kwargs)
    elif norm_type == "groupnorm":
        return tfa.layers.GroupNormalization(**kwargs)
    elif norm_type == "groupnorm_few_groups":
        return tfa.layers.GroupNormalization(epsilon=1e-5, groups=8)
    elif norm_type == "layernorm":
        return tf.keras.layers.LayerNormalization(**kwargs)
    elif norm_type == "layernorm_small_epsilon":
        return tf.keras.layers.LayerNormalization(epsilon=1e-6, **kwargs)
    elif norm_type == "syncbatchnorm":
        return tf.keras.layers.experimental.SyncBatchNormalization(**kwargs)
    elif norm_type == "syncbatchnorm_small_momentum":
        return tf.keras.layers.experimental.SyncBatchNormalization(
            momentum=.9, epsilon=0.01, **kwargs
        )
    elif not norm_type:
        return IdentityLayer()
    else:
        raise ValueError()

class IdentityLayer(tf.keras.layers.Layer):
    def call(self, inputs, training):
        return tf.identity(inputs)


# from https://github.com/naumanjaved/genformer_public/blob/128230bc07144b1bf314f872893b1be2f3c26539/enformer_vanilla/layers.py
@tf.keras.utils.register_keras_serializable()
class SoftmaxPooling1D(tf.keras.layers.Layer):
    def __init__(
        self,
        pool_size: int = 2,
        w_init_scale: float = 2.0,
        k_init=None,
        train=True,
        per_channel: bool = True,
        name: str = "SoftmaxPooling1D",
    ):
        """Softmax pooling from enformer
        Args:
          pool_size: Pooling size, same as in Max/AvgPooling.
          per_channel: If True, the logits/softmax weights will be computed for
            each channel separately. If False, same weights will be used across all
            channels.
          w_init_scale: When 0.0 is equivalent to avg pooling, and when
            ~2.0 and `per_channel=False` it's equivalent to max pooling.
          name: Module name.
        """
        super().__init__(name=name)
        self._pool_size = pool_size
        self._per_channel = per_channel
        self._w_init_scale = w_init_scale
        self._logit_linear = None
        self.train = train
        self._k_init = k_init

    def build(self, input_shape):
        num_features = input_shape[-1]
        if self._per_channel:
            units = num_features
        else:
            units = 1
        self._logit_linear = tf.keras.layers.Dense(
            units=units,
            use_bias=False,
            trainable=self.train,
            kernel_initializer=self._k_init
            if (self._k_init is not None)
            else tf.keras.initializers.Identity(gain=self._w_init_scale),
        )
        super(SoftmaxPooling1D, self).build(input_shape)

    ### revisit
    def call(self, inputs):
        _, length, num_features = inputs.shape
        # print(inputs.shape)
        inputs = tf.reshape(
            inputs, (-1, length // self._pool_size, self._pool_size, num_features)
        )
        out = tf.reduce_sum(
            inputs * tf.nn.softmax(self._logit_linear(inputs), axis=-2), axis=-2
        )
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "pool_size": self._pool_size,
                "w_init_scale": self._w_init_scale,
                "per_channel": self._per_channel,
            }
        )
        return config
