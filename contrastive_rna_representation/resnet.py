import tensorflow as tf

from contrastive_rna_representation.saluki_layers import StochasticShift
from contrastive_rna_representation.bpnet_dilated_conv import (
    make_dilated_basic_block_layer,
    make_dilated_bottleneck_block_layer,
    return_norm_layer,
    IdentityLayer,
)


class DilatedResnet(tf.keras.Model):
    def __init__(
        self,
        layer_params,
        num_classes,
        dilation_params=(1, 4, 16, 64),
        dropout_prob=0,
        max_pool='max_pool',
        kernel_size=8,
        filter_nums=(64, 128, 128, 256),
        norm_type="batchnorm",
        kernel_initializer="glorot_uniform",
        l2_scale=0,
        projection_l2_scale=0,
        pooling_layer="avgpool",
        projection_head_size=128,
        projection_body=512,
        increase_dilation=True,
        fc_head="",
        add_shift=False,
    ):
        super(DilatedResnet, self).__init__()
        self.layer_params = layer_params
        self.num_classes = num_classes
        self.dilation_params = dilation_params
        self.dropout_prob = dropout_prob
        self.max_pool = max_pool
        self.kernel_size = kernel_size
        self.filter_nums = filter_nums
        self.norm_type = norm_type
        self.kernel_initializer = kernel_initializer
        self.l2_scale = l2_scale
        self.pooling_layer = pooling_layer
        self.projection_head_size = projection_head_size
        self.projection_body = projection_body
        self.increase_dilation = increase_dilation
        self.fc_head = fc_head

        if add_shift:
            self.shift = StochasticShift(3, symmetric=False)
        else:
            self.shift = IdentityLayer()

        self.conv1 = tf.keras.layers.Conv1D(
            filters=filter_nums[0],
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer=kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
        )
        self.bn1 = return_norm_layer(norm_type)
        self.pool1 = tf.keras.layers.MaxPool1D(pool_size=2, padding="same")

        # dilation will be 2, 4
        self.layer1 = make_dilated_basic_block_layer(
            filter_num=filter_nums[0],
            kernel_size=kernel_size,
            dropout_prob=dropout_prob,
            dilation=dilation_params[0],
            max_pool=max_pool,
            blocks=layer_params[0],
            norm_type=norm_type,
            kernel_initializer=kernel_initializer,
            l2_scale=l2_scale,
            increase_dilation=increase_dilation,
        )
        # dilation will be 2*2=4, 2*4=8
        self.layer2 = make_dilated_basic_block_layer(
            filter_num=filter_nums[1],
            kernel_size=kernel_size,
            dropout_prob=dropout_prob,
            dilation=dilation_params[1],
            max_pool=max_pool,
            blocks=layer_params[1],
            norm_type=norm_type,
            kernel_initializer=kernel_initializer,
            l2_scale=l2_scale,
            increase_dilation=increase_dilation,
        )
        # dilation will be 8 * 2 = 16, 8 * 4 = 32
        self.layer3 = make_dilated_basic_block_layer(
            filter_num=filter_nums[2],
            kernel_size=kernel_size,
            dropout_prob=dropout_prob,
            dilation=dilation_params[2],
            max_pool=max_pool,
            blocks=layer_params[2],
            norm_type=norm_type,
            kernel_initializer=kernel_initializer,
            l2_scale=l2_scale,
            increase_dilation=increase_dilation,
        )
        # dilation will be 16 * 2 = 32, 16*4 = 64
        self.layer4 = make_dilated_basic_block_layer(
            filter_num=filter_nums[3],
            kernel_size=kernel_size,
            dropout_prob=dropout_prob,
            dilation=dilation_params[3],
            max_pool=max_pool,
            blocks=layer_params[3],
            norm_type=norm_type,
            kernel_initializer=kernel_initializer,
            l2_scale=l2_scale,
            increase_dilation=increase_dilation,
        )

        body_layers = [
            self.shift,
            self.conv1,
            self.bn1,
            self.pool1,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ]
        self.body = tf.keras.Sequential(
            body_layers,
            name="conv_body",
        )

        if pooling_layer == "gru":
            self.avgpool = tf.keras.layers.GRU(
                filter_nums[-1],
                go_backwards=True,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            )
        elif pooling_layer == "gru_non_lin":
            self.avgpool = tf.keras.Sequential(
                [
                    tf.keras.layers.GRU(
                        filter_nums[-1],
                        go_backwards=True,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
                    ),
                    return_norm_layer(norm_type),
                    tf.keras.layers.Activation("relu"),
                ]
            )
        elif pooling_layer == "channel_conv":
            self.avgpool = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv1D(
                        3,
                        kernel_size=1,
                        padding="same",
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
                    ),
                    tf.keras.layers.Reshape((-1,)),
                ]
            )
        elif pooling_layer == "avgpool":
            self.avgpool = tf.keras.layers.GlobalAveragePooling1D()

        elif pooling_layer == "channel_avgpool":
            # will instead pool along the channel dimension
            self.avgpool = tf.keras.layers.GlobalAveragePooling1D(
                data_format="channels_first"
            )

        else:
            raise ValueError(f"Unknown pooling layer {pooling_layer}")

        if fc_head == "linear_zero_init":
            self.fc = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        units=num_classes, kernel_initializer="zero", dtype="float32"
                    ),
                ],
                name="fc",
            )
        elif fc_head == "linear":
            self.fc = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        units=num_classes,
                        kernel_initializer=kernel_initializer,
                        dtype="float32"
                    ),
                ],
                name="fc",
            )
        elif fc_head == "linear_sigmoid":
            self.fc = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        units=num_classes, kernel_initializer=kernel_initializer,
                        dtype="float32", activation='sigmoid'
                    ),
                ],
                name="fc",
            )
        elif fc_head == "linear_sigmoid_zero_init":
            self.fc = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        units=num_classes, kernel_initializer="zero", dtype="float32",
                        activation='sigmoid'
                    ),
                ],
                name="fc",
            )

        elif fc_head == 'fc_l2':
            self.fc = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        units=filter_nums[-1],
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                    ),
                    return_norm_layer(norm_type),
                    tf.keras.layers.Activation("relu"),
                    tf.keras.layers.Dense(
                        units=num_classes,
                        activation=None,
                        dtype="float32",
                        kernel_initializer="zero",
                        kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                    ),
                ],
                name="fc",
            )
        elif fc_head == 'fc_zero_init':
            self.fc = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        units=filter_nums[-1],
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
                    ),
                    return_norm_layer(norm_type),
                    tf.keras.layers.Activation("relu"),
                    tf.keras.layers.Dense(
                        units=num_classes,
                        activation=None,
                        dtype="float32",
                        kernel_initializer="zero",
                        kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
                    ),
                ],
                name="fc",
            )


        else:
            self.fc = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        units=filter_nums[-1],
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
                    ),
                    return_norm_layer(norm_type),
                    tf.keras.layers.Activation("relu"),
                    tf.keras.layers.Dense(
                        units=num_classes,
                        activation=None,
                        dtype="float32",
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
                    ),
                ],
                name="fc",
            )

        self.projection_head = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    projection_body,
                    kernel_regularizer=tf.keras.regularizers.l2(projection_l2_scale),
                ),
                return_norm_layer(norm_type),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.Dense(
                    projection_body,
                    kernel_regularizer=tf.keras.regularizers.l2(projection_l2_scale),
                ),
                return_norm_layer(norm_type),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.Dense(
                    projection_head_size,
                    dtype="float32",
                    use_bias=False,
                    kernel_regularizer=tf.keras.regularizers.l2(projection_l2_scale),
                ),
            ],
            name="projection_head",
        )

    def representation(self, inputs, training=None, mask=None):
        x = self.body(inputs, training=training)
        return x

    def supervised_inference(self, inputs, training=None, mask=None):
        features = self.representation(inputs, training=training)
        x = self.avgpool(features)
        output = self.fc(x, training=training)
        return output

    def call(self, inputs, training=None, mask=None):
        x = self.supervised_inference(inputs, training=training)
        return x

    def contrastive_inference(self, inputs, training=True, mask=None):
        features = self.representation(inputs, training, mask)
        x = self.avgpool(features)
        output = self.projection_head(x, training=training)
        return output

    def get_config(self):
        config = super().get_config()
        config['layer_params'] = self.layer_params
        config['num_classes'] = self.num_classes
        config['dilation_params'] = self.dilation_params
        config['dropout_prob'] = self.dropout_prob
        config['max_pool'] = self.max_pool
        config['kernel_size'] = self.kernel_size
        config['filter_nums'] = self.filter_nums
        config['norm_type'] = self.norm_type
        config['kernel_initializer'] = self.kernel_initializer
        config['l2_scale'] = self.l2_scale
        config['pooling_layer'] = self.pooling_layer
        config['projection_head_size'] = self.projection_head_size
        config['projection_body'] = self.projection_body
        config['increase_dilation'] = self.increase_dilation
        config['fc_head'] = self.fc_head
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



def dilated_medium(
    num_classes,
    dropout_prob=0.1,
    norm_type="batchnorm",
    dilation_params=(1, 4, 32, 256),
    kernel_initializer="glorot_uniform",
    l2_scale=0,
    pooling_layer="avgpool",
    kernel_size=3,
    projection_head_size=128,
    projection_body=512,
    fc_head="",
    max_pool='max_pool'
):
    return DilatedResnet(
        filter_nums=(64, 128, 256, 512),
        layer_params=(3, 4, 6, 3),
        num_classes=num_classes,
        dropout_prob=dropout_prob,
        dilation_params=dilation_params,
        kernel_initializer=kernel_initializer,
        l2_scale=l2_scale,
        pooling_layer=pooling_layer,
        norm_type=norm_type,
        kernel_size=kernel_size,
        projection_head_size=projection_head_size,
        projection_body=projection_body,
        fc_head=fc_head,
        max_pool=max_pool,
    )


def less_dilated_medium(
    num_classes,
    dropout_prob=0.1,
    norm_type="batchnorm",
    dilation_params=(1, 2, 4, 8),
    kernel_initializer="glorot_uniform",
    l2_scale=0,
    pooling_layer="avgpool",
    kernel_size=3,
    block_type="dilated",
    M=2,
    G=32,
    r=16,
    L=16,
    projection_head_size=128,
    projection_body=512,
    fc_head="",
    max_pool='max_pool',
    add_shift=True,
):
    return DilatedResnet(
        filter_nums=(64, 128, 256, 512),
        layer_params=(3, 4, 6, 3),
        num_classes=num_classes,
        dropout_prob=dropout_prob,
        dilation_params=dilation_params,
        kernel_initializer=kernel_initializer,
        l2_scale=l2_scale,
        pooling_layer=pooling_layer,
        norm_type=norm_type,
        kernel_size=kernel_size,
        block_type=block_type,
        M=M,
        G=G,
        r=r,
        L=L,
        projection_head_size=projection_head_size,
        projection_body=projection_body,
        fc_head=fc_head,
        max_pool=max_pool,
        add_shift=add_shift,
        increase_dilation=False,
    )

def less_dilated_medium2(
    num_classes,
    dropout_prob=0.1,
    norm_type="batchnorm",
    dilation_params=(1, 2, 4, 8),
    kernel_initializer="glorot_uniform",
    l2_scale=0,
    pooling_layer="avgpool",
    kernel_size=3,
    block_type="dilated_bottleneck",
    M=2,
    G=32,
    r=16,
    L=16,
    projection_head_size=128,
    projection_body=512,
    fc_head="",
    max_pool='max_pool',
    add_shift=True,
):
    return DilatedResnet(
        filter_nums=(64, 128, 256, 512),
        layer_params=(3, 4, 6, 3),
        num_classes=num_classes,
        dropout_prob=dropout_prob,
        dilation_params=dilation_params,
        kernel_initializer=kernel_initializer,
        l2_scale=l2_scale,
        pooling_layer=pooling_layer,
        norm_type=norm_type,
        kernel_size=kernel_size,
        block_type=block_type,
        M=M,
        G=G,
        r=r,
        L=L,
        projection_head_size=projection_head_size,
        projection_body=projection_body,
        fc_head=fc_head,
        max_pool=max_pool,
        add_shift=add_shift,
        increase_dilation=False,
    )



def not_dilated_medium(
    num_classes,
    dropout_prob=0.1,
    norm_type="batchnorm",
    dilation_params=(1, 1, 1, 1),
    kernel_initializer="glorot_uniform",
    l2_scale=0,
    pooling_layer="avgpool",
    kernel_size=3,
    block_type="dilated",
    M=2,
    G=32,
    r=16,
    L=16,
    projection_head_size=128,
    projection_body=512,
    fc_head="",
    max_pool='max_pool'
):
    return DilatedResnet(
        filter_nums=(64, 128, 256, 512),
        layer_params=(3, 4, 6, 3),
        num_classes=num_classes,
        dropout_prob=dropout_prob,
        dilation_params=dilation_params,
        kernel_initializer=kernel_initializer,
        l2_scale=l2_scale,
        pooling_layer=pooling_layer,
        norm_type=norm_type,
        kernel_size=kernel_size,
        block_type=block_type,
        M=M,
        G=G,
        r=r,
        L=L,
        projection_head_size=projection_head_size,
        projection_body=projection_body,
        fc_head=fc_head,
        max_pool=max_pool,
        increase_dilation=False
    )


def dilated_large(
    num_classes,
    dropout_prob=0.1,
    norm_type="batchnorm",
    dilation_params=(1, 4, 16, 64),
    kernel_initializer="glorot_uniform",
    l2_scale=0,
    pooling_layer="avgpool",
    kernel_size=3,
    block_type="dilated",
    M=2,
    G=32,
    r=16,
    L=16,
    projection_head_size=128,
    projection_body=512,
    fc_head="",
    max_pool='max_pool'
):
    return DilatedResnet(
        filter_nums=(128, 256, 512, 1024),
        layer_params=(3, 4, 6, 3),
        num_classes=num_classes,
        dropout_prob=dropout_prob,
        dilation_params=dilation_params,
        kernel_initializer=kernel_initializer,
        l2_scale=l2_scale,
        pooling_layer=pooling_layer,
        norm_type=norm_type,
        kernel_size=kernel_size,
        block_type=block_type,
        M=M,
        G=G,
        r=r,
        L=L,
        projection_head_size=projection_head_size,
        projection_body=projection_body,
        fc_head=fc_head,
        max_pool=max_pool,
    )


def dilated_small2(
    num_classes,
    dropout_prob=0.1,
    norm_type="batchnorm",
    dilation_params=(1, 4, 16, 64),
    kernel_initializer="glorot_uniform",
    l2_scale=0,
    pooling_layer="avgpool",
    kernel_size=3,
    projection_body=512,
    projection_head_size=128,
    fc_head="",
    max_pool='max_pool'
):
    return DilatedResnet(
        filter_nums=(64, 128, 256, 512),
        layer_params=(2, 2, 2, 2),
        num_classes=num_classes,
        dropout_prob=dropout_prob,
        dilation_params=dilation_params,
        kernel_initializer=kernel_initializer,
        l2_scale=l2_scale,
        pooling_layer=pooling_layer,
        norm_type=norm_type,
        kernel_size=kernel_size,
        projection_head_size=projection_head_size,
        projection_body=projection_body,
        fc_head=fc_head,
        max_pool=max_pool,
    )


def dilated_small(
    num_classes,
    dropout_prob=0.1,
    norm_type="batchnorm",
    dilation_params=(1, 4, 16, 64),
    kernel_initializer="glorot_uniform",
    l2_scale=0,
    pooling_layer="avgpool",
    kernel_size=3,
    projection_body=512,
    projection_head_size=128,
    fc_head="",
    max_pool='max_pool',
    add_shift=True,
):
    return DilatedResnet(
        filter_nums=(64, 128, 128, 256),
        layer_params=(2, 2, 2, 2),
        num_classes=num_classes,
        dropout_prob=dropout_prob,
        dilation_params=dilation_params,
        kernel_initializer=kernel_initializer,
        l2_scale=l2_scale,
        pooling_layer=pooling_layer,
        norm_type=norm_type,
        kernel_size=kernel_size,
        projection_head_size=projection_head_size,
        projection_body=projection_body,
        fc_head=fc_head,
        max_pool=max_pool,
        add_shift=add_shift,
    )


def less_dilated_small(
    num_classes,
    dropout_prob=0.1,
    norm_type="batchnorm",
    dilation_params=(1, 2, 4, 8),
    kernel_initializer="glorot_uniform",
    l2_scale=0,
    pooling_layer="avgpool",
    kernel_size=3,
    block_type="dilated",
    M=2,
    G=32,
    r=16,
    L=16,
    projection_body=512,
    projection_head_size=128,
    fc_head="",
    max_pool='max_pool',
    add_shift=True,
):
    return DilatedResnet(
        filter_nums=(64, 128, 128, 256),
        layer_params=(2, 2, 2, 2),
        num_classes=num_classes,
        dropout_prob=dropout_prob,
        dilation_params=dilation_params,
        kernel_initializer=kernel_initializer,
        l2_scale=l2_scale,
        pooling_layer=pooling_layer,
        norm_type=norm_type,
        kernel_size=kernel_size,
        block_type=block_type,
        M=M,
        G=G,
        r=r,
        L=L,
        projection_head_size=projection_head_size,
        projection_body=projection_body,
        fc_head=fc_head,
        max_pool=max_pool,
        increase_dilation=True,
        add_shift=add_shift,
    )

def less_dilated_small2(
    num_classes,
    dropout_prob=0.1,
    norm_type="batchnorm",
    dilation_params=(1, 2, 4, 8),
    kernel_initializer="glorot_uniform",
    l2_scale=0,
    pooling_layer="avgpool",
    kernel_size=3,
    block_type="dilated",
    M=2,
    G=32,
    r=16,
    L=16,
    projection_body=512,
    projection_head_size=128,
    fc_head="",
    max_pool='max_pool',
    add_shift=True,
):
    return DilatedResnet(
        filter_nums=(64, 128, 128, 256),
        layer_params=(2, 2, 2, 2),
        num_classes=num_classes,
        dropout_prob=dropout_prob,
        dilation_params=dilation_params,
        kernel_initializer=kernel_initializer,
        l2_scale=l2_scale,
        pooling_layer=pooling_layer,
        norm_type=norm_type,
        kernel_size=kernel_size,
        block_type=block_type,
        M=M,
        G=G,
        r=r,
        L=L,
        projection_head_size=projection_head_size,
        projection_body=projection_body,
        fc_head=fc_head,
        max_pool=max_pool,
        increase_dilation=False,
        add_shift=add_shift,
    )

def dilated_med_small(
    num_classes,
    dropout_prob=0.1,
    norm_type="batchnorm",
    dilation_params=(1, 2, 4, 8),
    kernel_initializer="glorot_uniform",
    l2_scale=0,
    pooling_layer="avgpool",
    kernel_size=3,
    block_type="dilated",
    M=2,
    G=32,
    r=16,
    L=16,
    projection_body=512,
    projection_head_size=128,
    fc_head="",
    max_pool='max_pool',
    add_shift=True,
):
    return DilatedResnet(
        filter_nums=(64, 128, 256, 512),
        layer_params=(2, 2, 2, 2),
        num_classes=num_classes,
        dropout_prob=dropout_prob,
        dilation_params=dilation_params,
        kernel_initializer=kernel_initializer,
        l2_scale=l2_scale,
        pooling_layer=pooling_layer,
        norm_type=norm_type,
        kernel_size=kernel_size,
        block_type=block_type,
        M=M,
        G=G,
        r=r,
        L=L,
        projection_head_size=projection_head_size,
        projection_body=projection_body,
        fc_head=fc_head,
        max_pool=max_pool,
        increase_dilation=False,
        add_shift=add_shift,
    )

def dilated_small_constant_filters(
    num_classes,
    dropout_prob=0.1,
    norm_type="batchnorm",
    dilation_params=(1, 4, 16, 64),
    kernel_initializer="glorot_uniform",
    l2_scale=0,
    pooling_layer="avgpool",
    kernel_size=3,
    projection_body=512,
    projection_head_size=128,
    fc_head="",
    max_pool='max_pool',
):
    return DilatedResnet(
        filter_nums=(128, 128, 128, 128),
        layer_params=(2, 2, 2, 2),
        num_classes=num_classes,
        dropout_prob=dropout_prob,
        dilation_params=dilation_params,
        kernel_initializer=kernel_initializer,
        l2_scale=l2_scale,
        pooling_layer=pooling_layer,
        norm_type=norm_type,
        kernel_size=kernel_size,
        projection_body=projection_body,
        projection_head_size=projection_head_size,
        fc_head=fc_head,
        max_pool=max_pool,
    )


def not_dilated_small(
    num_classes,
    dropout_prob=0.1,
    norm_type="batchnorm",
    dilation_params=(1, 1, 1, 1),
    kernel_initializer="glorot_uniform",
    l2_scale=0,
    pooling_layer="avgpool",
    kernel_size=3,
    projection_body=512,
    projection_head_size=128,
    fc_head="",
    max_pool='max_pool',
):
    return DilatedResnet(
        filter_nums=(128, 128, 128, 128),
        layer_params=(2, 2, 2, 2),
        num_classes=num_classes,
        dropout_prob=dropout_prob,
        dilation_params=dilation_params,
        kernel_initializer=kernel_initializer,
        l2_scale=l2_scale,
        pooling_layer=pooling_layer,
        norm_type=norm_type,
        kernel_size=kernel_size,
        projection_head_size=projection_head_size,
        projection_body=projection_body,
        increase_dilation=False,
        fc_head=fc_head,
        max_pool=max_pool,
    )


def dilated_tiny(
    num_classes,
    dropout_prob=0.1,
    norm_type="batchnorm",
    dilation_params=(1, 4, 16, 64),
    kernel_initializer="glorot_uniform",
    l2_scale=0,
    pooling_layer="avgpool",
    kernel_size=3,
    projection_body=512,
    projection_head_size=128,
    fc_head="",
    max_pool='max_pool',
):
    return DilatedResnet(
        filter_nums=(32, 64, 64, 128),
        layer_params=(2, 2, 2, 2),
        num_classes=num_classes,
        dropout_prob=dropout_prob,
        dilation_params=dilation_params,
        kernel_initializer=kernel_initializer,
        l2_scale=l2_scale,
        pooling_layer=pooling_layer,
        norm_type=norm_type,
        kernel_size=kernel_size,
        projection_head_size=projection_head_size,
        projection_body=projection_body,
        fc_head=fc_head,
        max_pool=max_pool,
    )


def dilated_extra_tiny(
    num_classes,
    dropout_prob=0.1,
    norm_type="batchnorm",
    dilation_params=(1, 4, 16, 64),
    kernel_initializer="glorot_uniform",
    l2_scale=0,
    pooling_layer="avgpool",
    kernel_size=3,
    projection_head_size=128,
    fc_head="",
    projection_body=512,
    max_pool='max_pool',
):
    return DilatedResnet(
        filter_nums=(32, 32, 32, 64),
        layer_params=(2, 2, 2, 2),
        num_classes=num_classes,
        dropout_prob=dropout_prob,
        dilation_params=dilation_params,
        kernel_initializer=kernel_initializer,
        l2_scale=l2_scale,
        pooling_layer=pooling_layer,
        norm_type=norm_type,
        kernel_size=kernel_size,
        projection_head_size=projection_head_size,
        projection_body=projection_body,
        fc_head=fc_head,
        max_pool=max_pool,
    )
