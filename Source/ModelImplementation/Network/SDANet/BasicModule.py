# -*- coding: utf-8 -*-
from JackBasicStructLib.NN.Layer import *
from JackBasicStructLib.NN.Block import *


class ExtractUnaryFeatureModule(object):
    """docstring for ClassName"""

    def __init__(self, arg=None):
        super(ExtractUnaryFeatureModule, self).__init__()
        self.arg = arg

    def Inference(self, x, training=True):
        with tf.variable_scope("ExtractUnaryFeatureModule"):
            x = self.__ExtractUnaryFeatureBlock1(x, training=training)
            output_raw = self.__ExtractUnaryFeatureBlock2(x, training=training)
            output_skip_1 = self.__ExtractUnaryFeatureBlock3(output_raw, training=training)
            output_skip_2 = self.__ExtractUnaryFeatureBlock4(output_skip_1, training=training)
            x = ASPPBlock(output_skip_2, 32, "ASPP", training=training)
            x = tf.concat([output_raw, output_skip_1, output_skip_2, x], axis=3)
            x = self.__ExtractUnaryFeatureBlock5(x, training=training)
        return x

    def __ExtractUnaryFeatureBlock1(self, x, training=True):
        with tf.variable_scope("ExtractUnaryFeatureBlock1"):
            x = Conv2DLayer(x, 3, 2, 32, "Conv_1", training=training)
            x = Conv2DLayer(x, 3, 1, 32, "Conv_2", training=training)
            x = Conv2DLayer(x, 3, 1, 32, "Conv_3", training=training)

            res_block_num = 3
            for i in range(res_block_num):
                x = Res2DBlock(x, 3, "Res_" + str(i), training=training)

        return x

    def __ExtractUnaryFeatureBlock2(self, x, training=True):
        with tf.variable_scope("ExtractUnaryFeatureBlock2"):
            shortcut = Conv2DLayer(x, 3, 2, 64, "Conv_1", training=training)
            x = Conv2DLayer(x, 1, 2, 64, "Conv_w", training=training)
            x = tf.add(x, shortcut)

            res_block_num = 4
            for i in range(res_block_num):
                x = Bottleneck2DBlock(x, "BottleNeck_" + str(i), training=training)

        return x

    def __ExtractUnaryFeatureBlock3(self, x, training=True):
        with tf.variable_scope("ExtractUnaryFeatureBlock3"):
            x = Conv2DLayer(x, 3, 1, 128, "Conv_1", training=training)
            x = ResAtrousBlock(x, 3, 2, "Atrous_1", training=training)

        return x

    def __ExtractUnaryFeatureBlock4(self, x, training=True):
        with tf.variable_scope("ExtractUnaryFeatureBlock4"):
            x = Conv2DLayer(x, 3, 1, 128, "Conv_1", training=training)
            x = ResAtrousBlock(x, 3, 4, "Atrous_1", training=training)
        return x

    def __ExtractUnaryFeatureBlock5(self, x, training=True):
        with tf.variable_scope("ExtractUnaryFeatureBlock5"):
            x = Conv2DLayer(x, 3, 1, 32, "Conv_1", training=training)
            x = Conv2DLayer(x, 3, 1, 1, "Conv_2", biased=True,
                            bn=False, relu=False, training=training)
        return x
