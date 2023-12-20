import random
import os

import numpy as np
import tvm
import tvm.testing
from tvm import te, auto_scheduler, runtime, topi
from tvm.auto_scheduler import _ffi_api
from tvm.topi.utils import get_const_tuple
from tvm.topi.sparse.utils import random_bsr_matrix


class RandomConvOperator:
    def __init__(self,N=1, H=7, W=7, CO=64, CI=64, KH=3, KW=3, strides=(1, 1), padding=(1, 1)):
        self.params = {
            'N': N,
            'H': H,
            'W': W,
            'CO': CO,
            'CI': CI,
            'KH': KH,
            'KW': KW,
            'strides': strides,
            'padding': padding
        }
        self.randomize_params()

    def randomize_params(self):
        self.params['N'] = random.randint(1, 10)
        self.params['H'] = random.randint(1, 10)
        self.params['W'] = random.randint(1, 10)
        self.params['CO'] = random.randint(10, 100)
        self.params['CI'] = random.randint(10, 100)
        self.params['KH'] = random.randint(3,4)
        self.params['KW'] = random.randint(3,4)
        self.params['strides'] = (random.randint(1, 3), random.randint(1, 3))
        self.params['padding'] = (random.randint(1, 1), random.randint(1, 1))

    def get_param_values(self):
        return [value for value in self.params.values()]

class RandomMatmul:
    def __init__(self,batch=2, K=8,M=8,N=8):
        self.params = {
            'batch': batch,
            'K': K,
            'M': M,
            'N': N
        }
        self.randomize_params()

    def randomize_params(self):
        self.params['batch'] = random.randint(1, 10)
        self.params['K'] = random.randint(1, 10) 
        self.params['M'] = random.randint(1, 10) 
        self.params['N'] = random.randint(1, 10) 

    def get_param_values(self):
        return [value for value in self.params.values()]


class RandomNCHW:
    def __init__(self,N=1,C=3, H=8, W=8, ):
        self.params = {
            'N': N,
            'C': C,
            'H': H,
            'W': W
        }
        self.randomize_params()

    def randomize_params(self):
        self.params['N'] = random.randint(1, 10)
        self.params['C'] = random.randint(1, 10) * 4
        self.params['H'] = random.randint(1, 10) * 4
        self.params['W'] = random.randint(1, 10) * 4

    def get_param_values(self):
        return [value for value in self.params.values()]

# 使用示例
# conv_op = RandomConvOperator()
# conv_op.randomize_params()
# op_args = conv_op.get_param_values()

