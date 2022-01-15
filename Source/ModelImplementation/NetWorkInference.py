# -*- coding: utf-8 -*-
from Basic.Switch import Switch
from Basic.LogHandler import *
from Basic.Define import *
from JackBasicStructLib.Basic.Paras import *
from .Network.SDANet.Model import SDANet as SDA
from .Dataloader.Dataloader_SDA import DataHandler as Dataloader_SDA


class NetWorkInference(object):
    def __init__(self):
        pass

    def Inference(self, args, is_training=True):
        name = args.modelName
        for case in Switch(name):
            if case('SDANet'):
                Info("Begin loading SDA Model")
                paras = self.__Args2Paras(args, is_training)
                model = SDA(args, is_training)
                dataHandler = Dataloader_SDA(args)
                break
            if case():
                Error('NetWork Type Error!!!')

        return paras, model, dataHandler

    def __Args2Paras(self, args, is_training):
        paras = Paras(args.learningRate, args.batchSize,
                      args.gpu, args.imgNum,
                      args.valImgNum, args.maxEpochs,
                      args.log, args.modelDir,
                      MODEL_NAME, args.auto_save_num,
                      10, args.pretrain,
                      1, is_training)
        return paras
