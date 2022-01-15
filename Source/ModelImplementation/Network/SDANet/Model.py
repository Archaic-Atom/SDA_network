# -*- coding: utf-8 -*-
from Basic.Define import *
from .BasicModule import *
from Basic.LogHandler import *
from JackBasicStructLib.Model.Template.ModelTemplate import ModelTemplate
from Evaluation.Accuracy import *
from Evaluation.Loss import *
from Evaluation.Warping import *
from JackBasicStructLib.Evaluation.AdamBound import *
from JackBasicStructLib.FamousBlock.UNet import *
from JackBasicStructLib.FamousBlock.UNet_2 import *
import math
import datetime


class SDANet(ModelTemplate):
    def __init__(self, args, training=True):
        # the programs setting
        self.__args = args
        # input id
        self.input_edge_imgL_id = 0 #domain x
        self.input_edge_imgR_id = 1 #domain x
        self.input_rgb_imgL_id = 2  #domain x
        self.input_rgb_imgR_id = 3  #domain x
        self.rgb_imgL_id = 4         #domain y
        self.rgb_imgR_id = 5         #domain y
        self.input_edge_yL_id = 6    #domain y
        self.input_edge_yR_id = 7    #domain y

        # output 
        self.output_fake_rgb_imgL_x_id = 0     
        self.output_fake_rgb_imgR_x_id = 1    
        self.output_cycle_rgb_imgL_x_id = 2 
        self.output_cycle_rgb_imgR_x_id = 3
        self.output_fake_rgb_imgL_y_id =4
        self.output_fake_rgb_imgR_y_id =5
        self.output_cycle_rgb_imgL_y_id = 6
        self.output_cycle_rgb_imgR_y_id =7     
        self.output_ture_probailityL_x_id = 8
        self.output_ture_probailityR_x_id = 9
        self.output_fake_probailityL_x_id = 10
        self.output_fake_probailityR_x_id = 11
        self.output_ture_probailityL_y_id = 12
        self.output_ture_probailityR_y_id = 13
        self.output_fake_probailityL_y_id = 14
        self.output_fake_probailityR_y_id = 15
        self.output_same_rgb_imgL_x = 16
        self.output_same_rgb_imgR_x = 17
        self.output_same_rgb_imgL_y = 18
        self.output_same_rgb_imgR_y = 19


        # label id for driving datasets
        self.label_rgb_img_id = 0
        # the image size
        if training == True:
            self.height = args.corpedImgHeight
            self.width = args.corpedImgWidth
        else:
            self.height = args.padedImgHeight
            self.width = args.padedImgWidth

    def GenInputInterface(self):
        # the input
        input = []
        args = self.__args
        edge_imgL_x = tf.placeholder(tf.float32, shape=(
            args.batchSize * args.gpu, self.height, self.width, 1))
        edge_imgR_x = tf.placeholder(tf.float32, shape=(
            args.batchSize * args.gpu, self.height, self.width, 1))
        rgb_imgL_x = tf.placeholder(tf.float32, shape=(
            args.batchSize * args.gpu, self.height, self.width, 3))
        rgb_imgR_x = tf.placeholder(tf.float32, shape=(
            args.batchSize * args.gpu, self.height, self.width, 3))
        rgb_imgL_y = tf.placeholder(tf.float32, shape=(
            args.batchSize * args.gpu, self.height, self.width, 3))
        rgb_imgR_y = tf.placeholder(tf.float32, shape=(
            args.batchSize * args.gpu, self.height, self.width, 3))
        edge_imgL_y = tf.placeholder(tf.float32, shape=(
            args.batchSize * args.gpu, self.height, self.width, 1))
        edge_imgR_y = tf.placeholder(tf.float32, shape=(
            args.batchSize * args.gpu, self.height, self.width, 1))
        input.append(edge_imgL_x)
        input.append(edge_imgR_x)
        input.append(rgb_imgL_x)
        input.append(rgb_imgR_x)
        input.append(rgb_imgL_y)
        input.append(rgb_imgR_y)
        input.append(edge_imgL_y)
        input.append(edge_imgR_y)

        return input

    def GenLabelInterface(self):
        label = []
        args = self.__args

        rgb_img = tf.placeholder(tf.float32, shape=(
            args.batchSize * args.gpu, self.height, self.width))
        label.append(rgb_img)

        return label

    def Optimizer(self, lr):
        opt_D_x = tf.train.AdamOptimizer(learning_rate=0.05 * lr, beta1=0.5)
        opt_D_y = tf.train.AdamOptimizer(learning_rate=0.01 * lr, beta1=0.5)
        opt_G_x = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
        opt_G_y = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)

        return [opt_G_x, opt_G_y, opt_D_x, opt_D_y]

    def OptimizerVarList(self):
        var_list = (tf.trainable_variables()
                    + tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
        assert set(var_list) == set(tf.trainable_variables())

        var_list_G_x = [var for var in var_list if var.name.startswith('generator_x')]
        self.__Count(var_list_G_x, "generator_x:")

        var_list_G_y = [var for var in var_list if var.name.startswith('generator_y')]
        self.__Count(var_list_G_y, "generator_y:")

        var_list_D_x = [var for var in var_list if var.name.startswith('discriminator_x')]
        self.__Count(var_list_D_x, "discriminator_x:")

        var_list_D_y = [var for var in var_list if var.name.startswith('discriminator_y')]
        self.__Count(var_list_D_y, "discriminator_y:")

        var_list = [var_list_G_x, var_list_G_y, var_list_D_x, var_list_D_y]

        return var_list

    def Accuary(self, output, label):
        acc = []

        acc_1L_x = D_ACC(output[self.output_ture_probailityL_x_id])
        acc_1R_x = D_ACC(output[self.output_ture_probailityR_x_id])
        acc_1_x = (acc_1L_x + acc_1R_x) * 0.5

        acc_1L_y = D_ACC(output[self.output_ture_probailityL_y_id])
        acc_1R_y = D_ACC(output[self.output_ture_probailityR_y_id])
        acc_1_y = (acc_1L_y + acc_1R_y) * 0.5

        acc_2L_x = D_ACC(output[self.output_fake_probailityL_x_id])
        acc_2R_x = D_ACC(output[self.output_fake_probailityR_x_id])
        acc_2_x = (acc_2L_x + acc_2R_x) * 0.5

        acc_2L_y = D_ACC(output[self.output_fake_probailityL_y_id])
        acc_2R_y = D_ACC(output[self.output_fake_probailityR_y_id])
        acc_2_y = (acc_2L_y + acc_2R_y) * 0.5

        acc.append(acc_1_x)
        acc.append(acc_2_x)
        acc.append(acc_1_y)
        acc.append(acc_2_y)

        return acc

    def Loss(self, output, input, label):
        loss = []
        # Adversarial loss for Discrinminator
        loss_realL_D_x = Cross_Entropy_Sigmoid(
            output[self.output_ture_probailityL_x_id],
            tf.ones_like(output[self.output_ture_probailityL_x_id]))
        loss_realR_D_x = Cross_Entropy_Sigmoid(
            output[self.output_ture_probailityR_x_id],
            tf.ones_like(output[self.output_ture_probailityR_x_id]))
        loss_real_D_x = (loss_realL_D_x + loss_realR_D_x) * 0.5

        loss_realL_D_y = Cross_Entropy_Sigmoid(
            output[self.output_ture_probailityL_y_id],
            tf.ones_like(output[self.output_ture_probailityL_y_id]))
        loss_realR_D_y = Cross_Entropy_Sigmoid(
            output[self.output_ture_probailityR_y_id],
            tf.ones_like(output[self.output_ture_probailityR_y_id]))
        loss_real_D_y = (loss_realL_D_y + loss_realR_D_y) * 0.5

        loss_fakeL_D_x = Cross_Entropy_Sigmoid(
            output[self.output_fake_probailityL_x_id],
            tf.zeros_like(output[self.output_fake_probailityL_x_id]))
        loss_fakeR_D_x = Cross_Entropy_Sigmoid(
            output[self.output_fake_probailityR_x_id],
            tf.zeros_like(output[self.output_fake_probailityR_x_id]))
        loss_fake_D_x = (loss_fakeL_D_x + loss_fakeR_D_x) * 0.5
        loss_D_x = (loss_real_D_x + loss_fake_D_x) * 0.5

        loss_fakeL_D_y = Cross_Entropy_Sigmoid(
            output[self.output_fake_probailityL_y_id],
            tf.zeros_like(output[self.output_fake_probailityL_y_id]))
        loss_fakeR_D_y = Cross_Entropy_Sigmoid(
            output[self.output_fake_probailityR_y_id],
            tf.zeros_like(output[self.output_fake_probailityR_y_id]))
        loss_fake_D_y = (loss_fakeL_D_y + loss_fakeR_D_y) * 0.5
        loss_D_y = (loss_real_D_y + loss_fake_D_y) * 0.5

        # Adversarial loss for Generator
        loss_fakeL_G_x = Cross_Entropy_Sigmoid(
            output[self.output_fake_probailityL_x_id],
            tf.ones_like(output[self.output_fake_probailityL_x_id]))
        loss_fakeR_G_x = Cross_Entropy_Sigmoid(
            output[self.output_fake_probailityR_x_id],
            tf.ones_like(output[self.output_fake_probailityR_x_id]))
        loss_fake_G_x = (loss_fakeL_G_x + loss_fakeR_G_x) * 0.5

        loss_fakeL_G_y = Cross_Entropy_Sigmoid(
            output[self.output_fake_probailityL_y_id],
            tf.ones_like(output[self.output_fake_probailityL_y_id]))
        loss_fakeR_G_y = Cross_Entropy_Sigmoid(
            output[self.output_fake_probailityR_y_id],
            tf.ones_like(output[self.output_fake_probailityR_y_id]))
        loss_fake_G_y = (loss_fakeL_G_y + loss_fakeR_G_y) * 0.5

        # L1 loss for generator X
        loss_Ll1_x = tf.reduce_mean(
                        tf.abs(output[self.output_fake_rgb_imgL_x_id] - input[self.input_rgb_imgL_id]))
        loss_Rl1_x = tf.reduce_mean(
                        tf.abs(output[self.output_fake_rgb_imgR_x_id] - input[self.input_rgb_imgR_id]))
        loss_l1_x = (loss_Ll1_x + loss_Rl1_x) * 0.5

        # Cycle consistency loss
        loss_cycleL_y= tf.reduce_mean(
            tf.abs(output[self.output_cycle_rgb_imgL_y_id] - input[self.rgb_imgL_id]))
        loss_cycleR_y = tf.reduce_mean(
            tf.abs(output[self.output_cycle_rgb_imgR_y_id] - input[self.rgb_imgR_id]))
        loss_cycle_y = (loss_cycleL_y + loss_cycleR_y) * 0.5

        loss_cycleL_x = tf.reduce_mean(
            tf.abs(output[self.output_cycle_rgb_imgL_x_id] - input[self.input_rgb_imgL_id]))
        loss_cycleR_x = tf.reduce_mean(
            tf.abs(output[self.output_cycle_rgb_imgR_x_id] - input[self.input_rgb_imgR_id]))
        loss_cycle_x = (loss_cycleL_x + loss_cycleR_x) * 0.5
        loss_cycle_total = loss_cycle_x + loss_cycle_y
        
        # Identity loss
        loss_identityL_y= tf.reduce_mean(
            tf.abs(output[self.output_same_rgb_imgL_y] - input[self.rgb_imgL_id]))
        loss_identityR_y = tf.reduce_mean(
            tf.abs(output[self.output_same_rgb_imgR_y] - input[self.rgb_imgR_id]))
        loss_identity_y = (loss_identityL_y + loss_identityR_y) * 0.5

        loss_identityL_x= tf.reduce_mean(
            tf.abs(output[self.output_same_rgb_imgL_x] - input[self.input_rgb_imgL_id]))
        loss_identityR_x= tf.reduce_mean(
            tf.abs(output[self.output_same_rgb_imgR_x] - input[self.input_rgb_imgR_id]))
        loss_identity_x = (loss_identityL_x + loss_identityR_x) * 0.5
        
        # Warping loss
        loss_warp = Warping_Loss(output[self.output_fake_rgb_imgL_x_id], 
                                 output[self.output_fake_rgb_imgR_x_id], label[self.label_rgb_img_id])

        # Total Generator loss
        loss_G_x = loss_fake_G_x + 10 * loss_cycle_total + 5 * loss_identity_y + 3 * loss_l1_x + 1 * loss_warp
        loss_G_y = loss_fake_G_y + 10 * loss_cycle_total + 5 * loss_identity_x
        

        loss.append(loss_G_x)
        loss.append(loss_G_y)
        loss.append(loss_D_x)
        loss.append(loss_D_y)
        loss.append(loss_l1_x)
        loss.append(loss_identity_x)
        loss.append(loss_identity_y)
        loss.append(loss_cycle_total)
        loss.append(loss_warp)

        return loss

    # This is the Inference, and you must have it!
    def Inference(self, input, training=True):
        edgeL_x, edgeR_x, rgb_imgL_x, rgb_imgR_x, rgb_imgL_y, rgb_imgR_y, edgeL_y, edgeR_y = self.__GetVar(input)


        fake_rgb_imgL_x = self.__Generator_x(rgb_imgL_x, edgeL_x, training=training)
        fake_rgb_imgR_x = self.__Generator_x(rgb_imgR_x, edgeR_x, training=training)        

        cycle_rgb_imgL_x = self.__Generator_y(fake_rgb_imgL_x, edgeL_x, training=training)
        cycle_rgb_imgR_x = self.__Generator_y(fake_rgb_imgR_x, edgeR_x, training=training)
        fake_rgb_imgL_y = self.__Generator_y(rgb_imgL_y, edgeL_y, training=training)
        fake_rgb_imgR_y = self.__Generator_y(rgb_imgR_y, edgeR_y, training=training)
        if self.__args.phase != 'test':
            cycle_rgb_imgL_y = self.__Generator_x(fake_rgb_imgL_y, edgeL_y, training=training)
            cycle_rgb_imgR_y = self.__Generator_x(fake_rgb_imgR_y, edgeR_y, training=training)
            same_rgb_imgL_x = self.__Generator_y(rgb_imgL_x, edgeL_x, training=training)
            same_rgb_imgR_x = self.__Generator_y(rgb_imgR_x, edgeR_x, training=training)
            same_rgb_imgL_y = self.__Generator_x(rgb_imgL_y, edgeL_y, training=training)
            same_rgb_imgR_y = self.__Generator_x(rgb_imgR_y, edgeR_y, training=training)
            with tf.variable_scope("discriminator_x") as scope:
                t_probL_x, f_probL_x = self.__NetWork_D(rgb_imgL_y, fake_rgb_imgL_x, training=training)
                scope.reuse_variables
                t_probR_x, f_probR_x = self.__NetWork_D(rgb_imgR_y, fake_rgb_imgR_x, training=training)
            with tf.variable_scope("discriminator_y") as scope:
                t_probL_y, f_probL_y = self.__NetWork_D(rgb_imgL_x, fake_rgb_imgL_y, training=training)
                scope.reuse_variables
                t_probR_y, f_probR_y = self.__NetWork_D(rgb_imgR_x, fake_rgb_imgR_y, training=training)
                
            output = self.__GenRes(fake_rgb_imgL_x, fake_rgb_imgR_x, cycle_rgb_imgL_x, cycle_rgb_imgR_x, 
                                fake_rgb_imgL_y, fake_rgb_imgR_y, cycle_rgb_imgL_y, cycle_rgb_imgR_y,
                                t_probL_x, t_probR_x, f_probL_x, f_probR_x, t_probL_y, t_probR_y, 
                                f_probL_y, f_probR_y, same_rgb_imgL_x, same_rgb_imgR_x, same_rgb_imgL_y, same_rgb_imgR_y)
        else:
            output = self.__GenRes(fake_rgb_imgL_x, fake_rgb_imgR_x, cycle_rgb_imgL_x, cycle_rgb_imgR_x, fake_rgb_imgL_y, fake_rgb_imgR_y)
        
        return output

    def __Count(self, var_list, info=""):
        total_parameters = 0
        for variable in var_list:
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1

            for dim in shape:
                variable_parameters *= dim.value

            total_parameters += variable_parameters

        info = info + ' The total parameter: %d' % total_parameters
        Info(info)

    def __NetWork_G(self, rgb_img, edge_img, training=True):
        with tf.variable_scope("UNet_G"):
            fake_rgb_img = UNet().Inference(rgb_img, edge_img, "unet_1",training=training)
            fake_rgb_img = tf.nn.tanh(fake_rgb_img)

        return fake_rgb_img


    def __Generator_x(self, input_img, edge_img, training=True):
        with tf.variable_scope("generator_x",  reuse=tf.AUTO_REUSE):
            Info('├── Begin Build Generator_x')
            fake_rgb_img = self.__NetWork_G(input_img, edge_img, training=training)
            Info('│   └── After Generator_x:' + str(fake_rgb_img.get_shape()))

        return fake_rgb_img
            
    def __Generator_y(self, input_img, edge_img, training=True):
        with tf.variable_scope("generator_y",  reuse=tf.AUTO_REUSE):
            Info('├── Begin Build Generator_y')
            fake_rgb_img = self.__NetWork_G(input_img, edge_img, training=training)
            Info('│   └── After Generator_y:' + str(fake_rgb_img.get_shape()))

        return fake_rgb_img
  
    def __NetWork_D(self, rgb_img, fake_rgb_img, training=True):
        with tf.variable_scope("UNet_D", reuse=tf.AUTO_REUSE) as scope:
            t_prob = ExtractUnaryFeatureModule().Inference(rgb_img, training=training)
            scope.reuse_variables()
            f_prob = ExtractUnaryFeatureModule().Inference(fake_rgb_img, training=training)

        return t_prob, f_prob

    def __GetVar(self, input):
        return input[self.input_edge_imgL_id], input[self.input_edge_imgR_id], input[self.input_rgb_imgL_id],input[self.input_rgb_imgR_id], input[self.rgb_imgL_id], input[self.rgb_imgR_id], input[self.input_edge_yL_id], input[self.input_edge_yR_id]

    def __GenRes(self, fake_rgb_imgL_x, fake_rgb_imgR_x, cycle_rgb_imgL_x, cycle_rgb_imgR_x, 
                fake_rgb_imgL_y, fake_rgb_imgR_y, cycle_rgb_imgL_y= None, cycle_rgb_imgR_y= None,
                t_probL_x= None, t_probR_x= None, f_probL_x= None, f_probR_x= None, t_probL_y= None, t_probR_y= None, 
                f_probL_y= None, f_probR_y= None, same_rgb_imgL_x= None, same_rgb_imgR_x= None, same_rgb_imgL_y= None, same_rgb_imgR_y= None):
        res = []
        res.append(fake_rgb_imgL_x)
        res.append(fake_rgb_imgR_x)
        
        res.append(cycle_rgb_imgL_x)
        res.append(cycle_rgb_imgR_x)
        res.append(fake_rgb_imgL_y)
        res.append(fake_rgb_imgR_y)
        if self.__args.phase != 'test':
            res.append(cycle_rgb_imgL_y)
            res.append(cycle_rgb_imgR_y)
            res.append(t_probL_x)
            res.append(t_probR_x)
            res.append(f_probL_x)
            res.append(f_probR_x)
            res.append(t_probL_y)
            res.append(t_probR_y)
            res.append(f_probL_y)
            res.append(f_probR_y)
            res.append(same_rgb_imgL_x)
            res.append(same_rgb_imgR_x)
            res.append(same_rgb_imgL_y)
            res.append(same_rgb_imgR_y)
        return res
