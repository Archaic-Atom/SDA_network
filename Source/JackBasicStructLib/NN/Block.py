# -*- coding: utf-8 -*-
from .Layer import *


def Res2DBlock(x, ksize, name, training=True):
    with tf.variable_scope(name + "/Res2DBlock"):
        filters_out = x.get_shape()[-1]
        shortcut = x

        x = Conv2DLayer(x, ksize, 1, filters_out, "ConvA", training=training)
        x = Conv2DLayer(x, ksize, 1, filters_out, "ConvB", relu=False, training=training)

        x = tf.add(shortcut, x)
        x = tf.nn.relu(x)

    return x


def Res3DBlock(x, ksize, name, training=True):
    with tf.variable_scope(name + "/Res3DBlock"):
        filters_out = x.get_shape()[-1]
        shortcut = x

        x = Conv3DLayer(x, ksize, 1, filters_out, "ConvA", training=training)
        x = Conv3DLayer(x, ksize, 1, filters_out, "ConvB", relu=False, training=training)

        x = tf.add(shortcut, x)
        x = tf.nn.relu(x)

    return x


def Bottleneck2DBlock(x,  name, training=True):
    with tf.variable_scope(name + "/Bottleneck2DBlock"):
        shortcutFeatureNum = x.get_shape()[-1]
        filters_out = shortcutFeatureNum // 2
        shortcut = x

        x = Conv2DLayer(x, 1, 1, filters_out, "ConvA", training=training)
        x = Conv2DLayer(x, 3, 1, filters_out, "ConvB", training=training)
        x = Conv2DLayer(x, 1, 1, shortcutFeatureNum, "ConvC", relu=False, training=training)

        x = tf.add(x, shortcut)
        x = tf.nn.relu(x)

    return x


def ResAtrousBlock(x, ksize, rate, name, training=True):
    with tf.variable_scope(name + "/ResAtrousBlock"):
        filters_out = x.get_shape()[-1]
        shortcut = x

        x = AtrousConv2DLayer(x, ksize, rate, filters_out, "ConvA", training=training)
        x = AtrousConv2DLayer(x, ksize, rate, filters_out, "ConvB", relu=False, training=training)

        x = tf.add(x, shortcut)
        x = tf.nn.relu(x)

    return x


def DownSamplingBlock(x, ksize, filters_out, name, training=True):
    with tf.variable_scope(name + "/DownSamplingBlock"):
        x = Conv3DLayer(x, ksize, 2, filters_out, "ConvA", training=training)
        x = Res3DBlock(x, ksize, "Res_1", training=training)
    return x


def UpSamplingBlock(x, ksize, filters_out, name, training=True):
    with tf.variable_scope(name + "/UpSamplingBlock"):
        x = DeConv3DLayer(x, ksize, 2, filters_out, "ConvA", training=training)
        x = Res3DBlock(x, ksize, "Res_1", training=training)
    return x


def SPPBlock(x, filters_out, name, training=True):
    with tf.variable_scope(name + "/SPP"):
        branch0 = AvgPoolingConv2DUpsampleLayer(
            x, 8, 8, filters_out, "Branch0", training=training)
        branch1 = AvgPoolingConv2DUpsampleLayer(
            x, 16, 16, filters_out, "Branch1", training=training)
        branch2 = AvgPoolingConv2DUpsampleLayer(
            x, 32, 32, filters_out, "Branch2", training=training)
        branch3 = AvgPoolingConv2DUpsampleLayer(
            x, 64, 64, filters_out, "Branch3", training=training)
        x = tf.concat([branch0, branch1, branch2, branch3], axis=-1)

    return x


def ASPPBlock(x, filters_out, name, training=True):
    with tf.variable_scope(name + "/ASPP"):
        branch0 = Conv2DLayer(x, 1, 1, filters_out, "branch0", training=training)
        branch1 = AtrousConv2DLayer(x, 3, 6, filters_out, "Branch1", training=training)
        branch2 = AtrousConv2DLayer(x, 3, 12, filters_out, "Branch2", training=training)
        branch3 = AtrousConv2DLayer(x, 3, 18, filters_out, "Branch3", training=training)
        x = tf.concat([branch0, branch1, branch2, branch3], axis=-1)
    return x


def SpaceTimeNonlocalBlock(x, name, training=True):
    with tf.variable_scope(name + "/SpaceTimeNonlocal"):
        batchsize, imgNum, height, width, shortcutFeatureNum = x.get_shape().as_list()
        filters_out = shortcutFeatureNum // 2

        with tf.variable_scope("p"):
            # Theta
            theta = Conv3DLayer(x, 1, 1, filters_out, "SpacetimeNonlocalTheta",
                                False, False, False, training)
            phi = Conv3DLayer(x, 1, 1, filters_out, "SpacetimeNonlocalPhi",
                              False, False, False, training)

            theta = tf.transpose(theta, [0, 1, 2, 3, 4])
            theta = tf.reshape(theta, shape=(batchsize, -1, filters_out))

            phi = tf.transpose(phi, [0, 1, 2, 3, 4])
            phi = tf.reshape(phi, shape=(batchsize, -1, filters_out))
            phi = tf.transpose(phi, [0, 2, 1])

            theta = tf.matmul(theta, phi)
            theta = tf.reshape(theta, shape=(batchsize, -1))
            theta = tf.nn.softmax(theta, axis=-1)
            theta = tf.reshape(theta, shape=(batchsize,
                                             imgNum*height*width, imgNum*height*width))
        with tf.variable_scope("g"):
            g = Conv3DLayer(x, 1, 1, filters_out, "SpacetimeNonlocalG",
                            False, False, False, training)
            g = tf.transpose(g, [0, 1, 2, 3, 4])
            g = tf.reshape(g, shape=(batchsize, -1, filters_out))

        with tf.variable_scope("g_p"):
            theta = tf.matmul(theta, g)
            theta = tf.reshape(theta, shape=(batchsize, filters_out,
                                             imgNum, height, width))
            theta = tf.transpose(theta, [0, 2, 3, 4, 1])

        theta = Conv3DLayer(theta, 1, 1, shortcutFeatureNum, "SpacetimeNonlocalConvBN",
                            False, True, False, training)
        x = tf.add(x, theta)

    return x


def GCBlock(x, name, r=16, training=True):
    with tf.variable_scope(name + "/GCBlock"):
        batchsize, height, width, shortcutFeatureNum = x.get_shape().as_list()
        with tf.variable_scope("k"):
            k = Conv2DLayer(x, 1, 1, 1, "GCblockK", False, False, False, training)
            k = tf.reshape(k, shape=(batchsize, -1, 1))
            k = tf.nn.softmax(k, axis=1)

        with tf.variable_scope("Theta"):
            theta = tf.reshape(x, shape=(batchsize, -1, shortcutFeatureNum))
            theta = tf.transpose(theta, [0, 2, 1])
            theta = tf.matmul(theta, k)
            theta = tf.reshape(theta, shape=(batchsize, 1, 1, shortcutFeatureNum))

            theta = Conv2DLayer(theta, 1, 1, shortcutFeatureNum // r,
                                "GCblockTheta", False, False, False, training)
            theta = LayerNorm(theta, "LayerNorm")
            theta = tf.nn.relu(theta)
            theta = Conv2DLayer(theta, 1, 1, shortcutFeatureNum,
                                "GCblockSE", False, False, False, training)
            theta = tf.tile(theta, [1, height, width, 1])

        x = tf.add(x, theta)

    return x


class GRUCell(tf.nn.rnn_cell.RNNCell):
    """docstring for ClassName"""

    def __init__(self, shape, ksize, filters_out, reuse=True):
        super(GRUCell, self).__init__()
        self._ksize = ksize
        self._filters_out = filters_out
        self._shape = tf.TensorShape(shape + [self._filters_out])
        self._reuse = reuse

    @property
    def state_size(self):
        return self._shape

    @property
    def output_size(self):
        return self._shape

    def __call__(self, x, state, name, training=True):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            with tf.variable_scope('Gates'):
                inputs = tf.concat([x, state], axis=-1)
                conv = Conv2DLayer(inputs, self._ksize, 1, 2*self._filters_out,
                                   "Conv_0", False, True, False, training)
                reset_gate, update_gate = tf.split(conv, 2, axis=-1)
                reset_gate = tf.sigmoid(reset_gate)
                update_gate = tf.sigmoid(update_gate)

            with tf.variable_scope('Output'):
                # concatenation
                inputs = tf.concat([x, reset_gate * state], axis=-1)
                conv = Conv2DLayer(inputs, self._ksize, 1, self._filters_out,
                                   "Conv_0", False, True, False, training)
                conv = tf.tanh(conv)
                x = update_gate * state + (1 - update_gate) * conv

        return x, x
