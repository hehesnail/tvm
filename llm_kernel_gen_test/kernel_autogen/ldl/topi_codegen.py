import os
import numpy as np
import tvm
import tvm.testing
from tvm import te, auto_scheduler, runtime, topi
from tvm.auto_scheduler import _ffi_api
from tvm.topi.utils import get_const_tuple
from tvm.topi.sparse.utils import random_bsr_matrix
import json
from nnop_random import RandomNCHW,RandomConvOperator,RandomMatmul, RandomNCHW32

from nn_codegen import Codegen
import nn_codegen


@auto_scheduler.register_workload
def sum(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.sum(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def cosh(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.cosh(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def cos(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.cos(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def acos(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.acos(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def asin(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.asin(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def asinh(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.asinh(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def atan(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.atan(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def atanh(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.atanh(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def ceil(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.ceil(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def clip(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.clip(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def const_vector(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.const_vector(data)
    return [data,out]
    
    

    
@auto_scheduler.register_workload
def erf(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.erf(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def exp(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.exp(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def fast_erf(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.fast_erf(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def fast_exp(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.fast_exp(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def fast_tanh(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.fast_tanh(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def fixed_point_multiply(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.fixed_point_multiply(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def flip(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.flip(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def floor(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.floor(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def full_like(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.full_like(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def isnan(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.isnan(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def log(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.log(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def log10(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.log10(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def log2(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.log2(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def max(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.max(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def min(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.min(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def negative(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.negative(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def prod(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.prod(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def reinterpret(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.reinterpret(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def repeat(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.repeat(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def reshape(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.reshape(data)
    return [data,out]
    

    
@auto_scheduler.register_workload
def round(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.round(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def rsqrt(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.rsqrt(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def shape(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.shape(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def sigmoid(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.sigmoid(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def sign(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.sign(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def sin(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.sin(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def sinh(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.sinh(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def sqrt(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.sqrt(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def tan(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.tan(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def tanh(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.tanh(data)
    return [data,out]
    
    
@auto_scheduler.register_workload
def tile(N,C, H, W):
    data = te.placeholder((N,C,H,W), name="data", dtype="float32")
    out = topi.tile(data)
    return [data,out]
    
    

def write_topi_json_to_file(op_name, c_code, cuda_code, ir_code,op_shape,save_file='topi_data.json'):
    op_data = {
        'op_name': op_name,
        'c_code': c_code,
        'cuda_code': cuda_code,
        'ir_code': ir_code,
        'op_shape':op_shape
    }
    json_str = json.dumps(op_data)
    with open(save_file, 'a') as f:
        f.write(json_str + '\n')

success_count = 0
failure_count = 0
def save_code(codegen_test,op_origin_name, operation, op_args_generator,max_attempts=10):
    attempts = 0
    global success_count
    global failure_count
    while attempts < max_attempts:
        op_args_generator.randomize_params()

        op_args = op_args_generator.get_param_values()
        print(*op_args)
        try:
            c_code, ir_code = codegen_test.c_codegen(topi_ops=operation, op_args=op_args)
            cuda_code = codegen_test.cuda_codegen(topi_ops=operation, op_args=op_args)
            # op_name = op_origin_name + str(op_args).replace(",", "_").replace(" ", "_")
            op_name = op_origin_name
            write_topi_json_to_file(op_name, c_code, cuda_code, ir_code,op_shape = op_args)
            success_count += 1
            attempts += 1
        except Exception as e:
            print(f"Code generation failed: {e}")
            failure_count += 1
            attempts += 1
list_topi_nchw_random = [sum,cosh,cos,acos,asin,asinh,atan,atanh,ceil,clip,const_vector,const_vector,const_vector,const_vector,erf,exp,fast_erf,fast_exp,fast_tanh,fixed_point_multiply,flip,floor,full_like,isnan,log,log10,log2,max,min,negative,prod,reinterpret,repeat,reshape,reshape,round,rsqrt,shape,sigmoid,sign,sin,sinh,sqrt,tan,tanh,tile]

for ops_function in list_topi_nchw_random:
    op_args_generator = RandomNCHW()
    conv_codegen = Codegen()
    op_name = ops_function.__name__
    save_code(conv_codegen,op_name, ops_function,op_args_generator=op_args_generator)  #3. 调用函数












# topi.sum
# topi.cosh
# topi.cos
# topi.acos
# topi.asin
# topi.asinh
# topi.atan
# topi.atanh
# topi.ceil
# topi.clip
# topi.const_vector
# topi.const_vector
# topi.const_vector
# topi.const_vector
# topi.erf
# topi.exp
# topi.fast_erf
# topi.fast_exp
# topi.fast_tanh
# topi.fixed_point_multiply
# topi.flip
# topi.floor
# topi.full_like
# topi.isnan
# topi.log
# topi.log10
# topi.log2
# topi.max
# topi.min
# topi.negative
# topi.prod
# topi.reinterpret
# topi.repeat
# topi.reshape
# topi.reshape
# topi.round
# topi.rsqrt
# topi.shape
# topi.sigmoid
# topi.sign
# topi.sin
# topi.sinh
# topi.sqrt
# topi.tan
# topi.tanh
# topi.tile