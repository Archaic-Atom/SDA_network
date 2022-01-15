# -*- coding: utf-8 -*-
from JackBasicStructLib.NN.Layer import *
from JackBasicStructLib.NN.Block import *
from Basic.LogHandler import *


class UNet_2(object):
    """docstring for UNet_2"""

    def __init__(self, arg=None):
        super(UNet_2, self).__init__()
        self.arg = arg

    def Inference(self, x, name, training=True):
        with tf.variable_scope(name + "/U-Net_2"):
            Info('├── Begin Build U-Net_2')

            Info('├── Begin Build Level 1')
            level_1 = self.__Level_1(x, "level_0", training=training)
            Info('│   └── After Level 1:' + str(level_1.get_shape()))

            Info('├── Begin Build Level 2')
            level_2 = self.__DownSampling(level_1, 128, 'Level_2', training=training)
            Info('│   └── After Level 2:' + str(level_2.get_shape()))

            Info('├── Begin Build Level 3')
            level_3 = self.__DownSampling(level_2, 256, 'Level_3', training=training)
            #level_3 = self.__DownSampling(level_2, 128, 'Level_3', training=training)
            Info('│   └── After Level 3:' + str(level_3.get_shape()))

            Info('├── Begin Build Level 4')
            level_4 = self.__DownSampling(level_3, 512, 'Level_4', training=training)
            Info('│   └── After Level 4:' + str(level_4.get_shape()))

            Info('├── Begin Build Level 5')
            level_5 = self.__Level_5(level_4, 'Level_5', training=training)
            Info('│   └── After Level 5:' + str(level_5.get_shape()))

            Info('├── Begin Build De-Level 4')
            level_4 = self.__UpSampling(level_5, level_4, 512, 'DeLevel_4', training=training)
            Info('│   └── After De-Level 4:' + str(level_4.get_shape()))

            Info('├── Begin Build De-Level 3')
            level_3 = self.__UpSampling(level_4, level_3, 256, 'DeLevel_3', training=training)
            #level_3 = self.__UpSampling(level_4, level_3, 128, 'DeLevel_3', training=training)
            level_3 = tf.nn.dropout(level_3, 0.7)
            Info('│   └── After De-Level 3:' + str(level_3.get_shape()))

            Info('├── Begin Build De-Level 2')
            level_2 = self.__UpSampling(level_3, level_2, 128, 'DeLevel_2', training=training)
            Info('│   └── After De-Level 2:' + str(level_2.get_shape()))

            #Info('├── Begin Build De-Level 1')
            #level_1 = self.__UpSampling(level_2, level_1, 64, 'DeLevel_1', training=training)
            #Info('│   └── After De-Level 1:' + str(level_1.get_shape()))

            Info('├── Begin Build De-Level 1')
            x = self.__DeLebel_1(level_2, 3, "DeLevel_0", training=training)
            Info('│   └── After De-Level 1:' + str(x.get_shape()))

        return x

    def __Level_1(self, x, name, training=True):
        with tf.variable_scope(name):
            x = Conv2DLayer(x, 3, 1, 64, "Conv_1", training=training)
            x = Conv2DLayer(x, 3, 1, 64, "Conv_2", training=training)
        return x

    def __DownSampling(self, x, filters_out, name, training=True):
        with tf.variable_scope(name):
            #x = MaxPooling2D(x, 3, 2)
            x = Conv2DLayer(x, 3, 2, filters_out // 2, "Conv_0", training=training)
            x = Conv2DLayer(x, 3, 1, filters_out, "Conv_1", training=training)
            x = Conv2DLayer(x, 3, 1, filters_out, "Conv_2", training=training)

        return x

    def __Level_5(self, x, name, training=True):
        with tf.variable_scope(name):
            #x = MaxPooling2D(x, 3, 2)
            x = Conv2DLayer(x, 3, 2, 512, "Conv_0", training=training)
            x = Conv2DLayer(x, 3, 1, 1024, "Conv_1", training=training)
            x = DeConv2DLayer(x, 3, 2, 512, "Conv_2", training=training)

        return x

    def __UpSampling(self, x, shortcut, filters_out, name, training=True):
        with tf.variable_scope(name):
            x = tf.concat([x, shortcut], axis=3)
            x = Conv2DLayer(x, 3, 1, filters_out, "Conv_1", training=training)
            x = DeConv2DLayer(x, 3, 2, filters_out // 2, "DeConv_1", training=training)
        return x

    def __DeLebel_1(self, x, filters_out, name, training=True):
        with tf.variable_scope(name):
            x = Conv2DLayer(x, 1, 1, filters_out, "Conv_1", biased=True, 
                            bn=False, relu=False,training=training)
        return x
