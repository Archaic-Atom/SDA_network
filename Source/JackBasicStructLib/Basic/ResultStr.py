# -*- coding: utf-8 -*-
#
#


class ResultStr(object):
    """docstring for ResultStr"""

    def __init__(self, arg=None):
        super(ResultStr, self).__init__()
        self.arg = arg

    def TrainingResultStr(self, epoch, loss, acc, duration, training=True):
        loss_str = self.Loss2Str(loss, decimal_places=6)
        acc_str = self.Acc2Str(acc, decimal_places=6)

        training_state = ""
        if training:
            training_state = "[TrainProcess] "
        else:
            training_state = "[ValProcess] "

        info_str = training_state + "e: " + str(epoch) + ', ' +\
            loss_str + ', ' + acc_str + ' (%.3f s/epoch)' % duration

        return info_str

    def TrainingIntermediateResult(self, epoch, loss, acc):
        loss_str = self.Loss2Str(loss, decimal_places=3)
        acc_str = self.Acc2Str(acc, decimal_places=3)

        info_str = 'e: ' + str(epoch) + ', ' +\
            loss_str + ', ' + acc_str

        return info_str

    def Loss2Str(self, loss, info_str=None, decimal_places=3):
        if info_str == None:
            info_str = []
            info_str = self.__GenInfoStr("l", len(loss))

        res = self.__Data2Str(loss, info_str, decimal_places)

        return res

    def Acc2Str(self, acc, info_str=None, decimal_places=3):
        if info_str == None:
            info_str = []
            info_str = self.__GenInfoStr("a", len(acc))

        res = self.__Data2Str(acc, info_str, decimal_places)
        return res

    def __GenInfoStr(self, info_str, num):
        res = []
        for i in range(num):
            res.append(info_str + str(i))
        return res

    def __Data2Str(self, data, info_str, decimal_places):
        assert len(data) == len(info_str)
        res = ""
        char_interval = ", "
        for i in range(len(info_str)):
            res = res + info_str[i] + \
                (": %." + str(decimal_places) + "f") % data[i] + char_interval

        char_offset = len(char_interval)
        res = res[:len(res)-char_offset]
        return res
