# -*- coding: utf-8 -*-
from Basic.LogHandler import *
from JackBasicStructLib.Model.Template.DataHandlerTemplate import DataHandlerTemplate
from JackBasicStructLib.FileProc.FileHandler import *
from JackBasicStructLib.Dataloader.KittiFlyingDataloader_GAN import KittiFlyingDataloader as kfd
from JackBasicStructLib.ImgProc.ImgHandler import *
from JackBasicStructLib.Basic.ResultStr import ResultStr
import time
import cv2
import numpy as np

TRAIN_ACC_FILE = 'train_acc.csv'                        # acc file's name
TRAIN_LOSS_FILE = 'train_loss.csv'                      # loss file's name
VAL_LOSS_FILE = 'val_loss.csv'                          # val file's name
VAL_ACC_FILE = 'val_acc.csv'                            # val file's name
TEST_ACC_FILE = 'test_acc.csv'                          # test file's name


LOW_ACC = 0.4
HIGH_ACC = 0.55


class DataHandler(DataHandlerTemplate):
    """docstring for DataHandler"""

    def __init__(self, args):
        super(DataHandler, self).__init__()
        self.__args = args
        self.fd_train_acc, self.fd_train_loss, self.fd_val_acc,\
            self.fd_val_loss, self.fd_test_acc = self.__CreateResultFile(args)
        self.kfd = kfd()
        self.result_str = ResultStr()
        self.acc = 0.6
        self.lr = args.learningRate

    def GetTrainingData(self, paras, trainList, num):
        edge_imgLs, edge_imgRs, intput_rgb_imgLs, intput_rgb_imgRs, rgb_imgLs, rgb_imgRs, input_edge_y_Ls, input_edge_y_Rs, label_rgb_imgs = self.kfd.GetBatchImage(
            self.__args, trainList, num)
        input, label = self.__CreateRes(edge_imgLs, edge_imgRs, intput_rgb_imgLs, intput_rgb_imgRs, rgb_imgLs, rgb_imgRs, input_edge_y_Ls, input_edge_y_Rs, label_rgb_imgs)
        return input, label

    def GetValData(self, paras, valList, num):
        edge_imgs, intput_rgb_imgs, label_rgb_imgs = self.kfd.GetBatchImage(
            self.__args, valList, num, True)
        lr = self.__GenLearningRate()
        input, label = self.__CreateRes(edge_imgs, intput_rgb_imgs, label_rgb_imgs)
        return input, label

    def GetTestingData(self, paras, testList, num):
        edge_imgLs, edge_imgRs, imgLs, imgRs, rgb_imgLs, rgb_imgRs, edge_img_y_Ls, edge_img_y_Rs, top_pads, left_pads, names = self.kfd.GetBatchTestImage(
            self.__args, testList, num, True)
        input, _ = self.__CreateRes(edge_imgLs, edge_imgRs, imgLs, imgRs, rgb_imgLs, rgb_imgRs, edge_img_y_Ls, edge_img_y_Rs, None)
        supplement = self.__CreateSupplement(top_pads, left_pads, names)
        self.start_time = time.time()
        return input, supplement

    def ShowTrainingResult(self, epoch, loss, acc, duration):
        info_str = self.result_str.TrainingResultStr(epoch, loss, acc, duration, True)
        Info(info_str)
        OutputData(self.fd_train_acc, loss[0])
        OutputData(self.fd_train_loss, acc[1])

    def ShowValResult(self, epoch, loss, acc, duration):
        info_str = self.result_str.TrainingResultStr(epoch, loss, acc, duration, False)
        Info(info_str)
        OutputData(self.fd_val_acc, loss[0])
        OutputData(self.fd_val_loss, acc[1])

    def ShowIntermediateResult(self, epoch, loss, acc):
        info_str = self.result_str.TrainingIntermediateResult(epoch, loss, acc)
        return info_str


    def SaveResult(self, output, supplement, imgID, testNum):
        args = self.__args
        res = np.array(output)
        top_pads = supplement[0]
        left_pads = supplement[1]
        names = supplement[2]
        ttimes = time.time() - self.start_time

        for i in range(args.gpu):
            for j in range(args.batchSize):
                temRes1 = res[i, 0, j, :, :, :]
                #temRes2 = res[i, 1, j, :, :, :]

                top_pad = top_pads[i*args.batchSize+j]
                left_pad = left_pads[i*args.batchSize+j]
                temRes1 = self.kfd.CropTestImg(temRes1, top_pad, left_pad)
                #temRes2 = self.kfd.CropTestImg(temRes2, top_pad, left_pad)

                if args.dataset == "KITTI":
                    name = args.gpu*args.batchSize * \
                        imgID + i*args.batchSize + j
                    self.kfd.SaveKITTITestData(args,  "%06d_10",temRes1, name)
                    #self.kfd.SaveKITTITestData(args,  "%06d_10",temRes2, name)
                    #print()


    def SaveResult_1(self, output, supplement, imgID, testNum):
        args = self.__args
        res = np.array(output)
        top_pads = supplement[0]
        left_pads = supplement[1]
        names = supplement[2]
        ttimes = time.time() - self.start_time

        for i in range(args.gpu):
            for j in range(args.batchSize):
                temRes = res[i, 0, j, :, :, :]

                top_pad = top_pads[i*args.batchSize+j]
                left_pad = left_pads[i*args.batchSize+j]
                temRes = self.kfd.CropTestImg(temRes, top_pad, left_pad)

                if args.dataset == "KITTI":
                    name = args.gpu*args.batchSize * \
                        imgID + i*args.batchSize + j
                    self.kfd.SaveKITTITestData(args, temRes, name)
                elif args.dataset == "ETH3D":
                    name = names[i*args.batchSize+j]
                    self.kfd.SaveETH3DTestData(args, temRes, name, ttimes)

                elif args.dataset == "Middlebury":
                    name = names[i*args.batchSize+j]
                    self.kfd.SaveMiddleburyTestData(args, temRes, name, ttimes)

    def SaveResult_2(self, output, supplement, imgID, testNum):
        args = self.__args
        res = np.array(output)
        top_pads = supplement[0]
        left_pads = supplement[1]
        names = supplement[2]
        ttimes = time.time() - self.start_time

        for i in range(args.gpu):
            for j in range(args.batchSize):
                temRes = res[i, 1, j, :, :, :]

                top_pad = top_pads[i*args.batchSize+j]
                left_pad = left_pads[i*args.batchSize+j]
                temRes = self.kfd.CropTestImg(temRes, top_pad, left_pad)

                if args.dataset == "KITTI":
                    name = args.gpu*args.batchSize * \
                        imgID + i*args.batchSize + j
                    self.kfd.SaveKITTITestData(args, temRes, name)
                elif args.dataset == "ETH3D":
                    name = names[i*args.batchSize+j]
                    self.kfd.SaveETH3DTestData(args, temRes, name, ttimes)

                elif args.dataset == "Middlebury":
                    name = names[i*args.batchSize+j]
                    self.kfd.SaveMiddleburyTestData(args, temRes, name, ttimes)

    def __GenLearningRate(self):
        args = self.__args
        if self.acc > HIGH_ACC:
            self.lr = args.learningRate
        elif self.acc < LOW_ACC:
            self.lr = args.learningRate * args.learningRate * 0.0005

        lr = np.expand_dims(self.lr, axis=0)
        return lr

    def __CreateRes(self, edge_imgLs, edge_imgRs, intput_rgb_imgLs, intput_rgb_imgRs, rgb_imgLs, rgb_imgRs, input_edge_y_Ls, input_edge_y_Rs, label_rgb_imgs):
        input = []
        label = []
        input.append(edge_imgLs)
        input.append(edge_imgRs)
        input.append(intput_rgb_imgLs)
        input.append(intput_rgb_imgRs)
        input.append(rgb_imgLs)
        input.append(rgb_imgRs)
        input.append(input_edge_y_Ls)
        input.append(input_edge_y_Rs)
        label.append(label_rgb_imgs)
        return input, label




    def __CreateSupplement(self, top_pads, left_pads, names):
        supplement = []
        supplement.append(top_pads)
        supplement.append(left_pads)
        supplement.append(names)
        return supplement

    def __CreateResultFile(self, args):
        # create the dir
        Info("Begin create the result folder")
        Mkdir(args.outputDir)
        Mkdir(args.resultImgDir)

        fd_train_acc = OpenLogFile(args.outputDir + TRAIN_LOSS_FILE, args.pretrain)
        fd_train_loss = OpenLogFile(args.outputDir + TRAIN_ACC_FILE, args.pretrain)
        fd_val_acc = OpenLogFile(args.outputDir + VAL_ACC_FILE, args.pretrain)
        fd_val_loss = OpenLogFile(args.outputDir + VAL_LOSS_FILE, args.pretrain)
        fd_test_acc = OpenLogFile(args.outputDir + TEST_ACC_FILE, args.pretrain)

        Info("Finish create the result folder")
        return fd_train_acc, fd_train_loss, fd_val_acc, fd_val_loss, fd_test_acc
