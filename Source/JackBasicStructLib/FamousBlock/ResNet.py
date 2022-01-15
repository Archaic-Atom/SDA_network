# -*- coding: utf-8 -*-
from JackBasicStructLib.NN.Layer import *
from JackBasicStructLib.NN.Block import *
from Basic.LogHandler import *


class ResNet(object):
    """docstring for ResNet"""

    def __init__(self, arg=None):
        super(ResNet, self).__init__()
        self.arg = arg

    def Inference(self, x, training=True):
        with tf.variable_scope("ResNet101"):
            Info('├── Begin Build ResNet101')
            Info('├── Begin Build Conv1')
            res = []
            x = self.Conv1(x, training=training)
            Info('│   └── After Conv1:' + str(x.get_shape()))
            res.append(x)

            Info('├── Begin Build Conv2')
            x = self.Conv2(x, training=training)
            Info('│   └── After Conv2:' + str(x.get_shape()))
            res.append(x)

            Info('├── Begin Build Conv3')
            x = self.Conv3(x, training=training)
            Info('│   └── After Conv3:' + str(x.get_shape()))
            res.append(x)

            Info('├── Begin Build Conv4')
            x = self.Conv4(x, training=training)
            Info('│   └── After Conv2:' + str(x.get_shape()))
            res.append(x)

            Info('├── Begin Build Conv5')
            x = self.Conv5(x, training=training)
            Info('│   └── After Conv5:' + str(x.get_shape()))
            res.append(x)

        return res

    def Conv1(self, x, training=True):
        with tf.variable_scope("Conv1_x"):
            x = Conv2DLayer(x, 7, 2, 64, "Conv_1", training=training)
        return x

    def Conv2(self, x, training=True):
        with tf.variable_scope("Conv2_x"):
            x = MaxPooling2D(x, 3, 2)
            blocks_num = 3
            for i in range(blocks_num):
                x = Bottleneck2DBlock(x, 1, 256, "Conv_" + str(i), training=training)

        return x

    def Conv3(self, x, training=True):
        with tf.variable_scope("Conv3_3"):
            x = Bottleneck2DBlock(x, 2, 512, "Conv_0", training=training)
            blocks_num = 3
            for i in range(blocks_num):
                x = Bottleneck2DBlock(x, 1, 512, "Conv_" + str(i+1), training=training)

        return x

    def Conv4(self, x, training=True):
        with tf.variable_scope("Conv4_x"):
            x = Bottleneck2DBlock(x, 2, 1024, "Conv_0", training=training)
            blocks_num = 23
            for i in range(blocks_num):
                x = Bottleneck2DBlock(x, 1, 1024, "Conv_" + str(i+1), training=training)

        return x

    def Conv5(self, x, training=True):
        with tf.variable_scope("Conv5_x"):
            x = Bottleneck2DBlock(x, 2, 2048, "Conv_0", training=training)
            blocks_num = 2
            for i in range(blocks_num):
                x = Bottleneck2DBlock(x, 1, 2048, "Conv_" + str(i+1), training=training)

        return x
