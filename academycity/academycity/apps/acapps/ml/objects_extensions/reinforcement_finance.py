
from ..basic_ml_objects import BaseDataProcessing, BasePotentialAlgo

import numpy as np
import tensorflow as tf
import timeit

#
# https://github.com/Apress/reinforcement-learning-finance
# https://github.com/PacktPublishing/Deep-Reinforcement-Learning-with-Python
#
class RIFAlgo(object):
    def __init__(self, dic):
        # print("90567-888-0100 RIFAlgo\n", dic, '\n', '-'*50)
        try:
            super(RIFAlgo, self).__init__()
        except Exception as ex:
            print("Error 9057-888-11 RIFAlgo:\n"+str(ex), "\n", '-'*50)
        # print("MLAlgo\n", self.app)
        # print("90004-020 RIAlgo\n", dic, '\n', '-'*50)
        self.app = dic["app"]


class RIFDataProcessing(BaseDataProcessing, BasePotentialAlgo, RIFAlgo):
    def __init__(self, dic):
        # print("90567-888-1 RIFDataProcessing\n", dic, '\n', '-' * 50)
        super().__init__(dic)
        # print("9057-888-  RIFDataProcessing ", self.app)

    def test(self, dic):
        print("9057-999-1: \n", "="*50, "\n", dic, "\n", "="*50)

        price = np.full(3 * 2, 2, dtype=np.float32)
        print(price)
        price = np.cumprod(price)
        print(price)

        result = {"status": "ok"}
        return result

