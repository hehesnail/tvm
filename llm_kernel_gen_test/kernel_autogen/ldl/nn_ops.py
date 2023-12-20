import os
import numpy as np
import tvm
import tvm.testing
from tvm import te, auto_scheduler, runtime, topi
from tvm.auto_scheduler import _ffi_api
from tvm.topi.utils import get_const_tuple
from tvm.topi.sparse.utils import random_bsr_matrix
import json
from nnop_random import RandomNCHW

from nn_codegen import Codegen
import nn_codegen
@auto_scheduler.register_workload
def conv2d(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, bias, conv]

@auto_scheduler.register_workload
def adaptive_pool(N,C,H,W):
    data = te.placeholder((N,C,H,W), name='data')
    output_size = (8, 8)
    out = topi.nn.adaptive_pool(data, output_size, layout='NCHW', pool_type='avg')
    return [data,out]

@auto_scheduler.register_workload
def add(N,C,H,W):
    lhs = te.placeholder((N,C,H,W), name='lhs')
    rhs = te.placeholder((N,C,H,W), name='rhs')
    out = topi.nn.add(lhs,rhs)
    return [lhs,rhs,out]

@auto_scheduler.register_workload
def rms_norm(N,C,H,W):
    data = te.placeholder((N,C,H,W), name='data')
    weight = te.placeholder((N,C,H,W), name='weight')
    axis = [1,2,3]
    out = topi.nn.rms_norm(data,weight,axis)
    return [data,weight,axis,out]

    
def write_json_to_file(op_name, c_code, cuda_code, ir_code,save_file='data.json'):
    op_data = {
        'op_name': op_name,
        'c_code': c_code,
        'cuda_code': cuda_code,
        'ir_code': ir_code
    }
    json_str = json.dumps(op_data)
    with open(save_file, 'a') as f:
        f.write(json_str + '\n')

def save_code(codegen_test,op_origin_name, operation, op_args_generator,max_attempts=10):
    attempts = 0
    global success_count
    global failure_count
    while attempts < max_attempts:
        op_args_generator.randomize_params()
        op_args = op_args_generator.get_param_values()

        c_code, ir_code = codegen_test.c_codegen(topi_ops=operation, op_args=op_args)
        cuda_code = codegen_test.cuda_codegen(topi_ops=operation, op_args=op_args)
        op_name = op_origin_name + str(op_args).replace(",", "_").replace(" ", "_")
        write_json_to_file(op_name, c_code, cuda_code, ir_code)
        success_count += 1
        attempts += 1

        # try:
        #     c_code, ir_code = codegen_test.c_codegen(topi_ops=operation, op_args=op_args)
        #     cuda_code = codegen_test.cuda_codegen(topi_ops=operation, op_args=op_args)
        #     op_name = op_origin_name + str(op_args).replace(",", "_").replace(" ", "_")
        #     write_json_to_file(op_name, c_code, cuda_code, ir_code)
        #     success_count += 1
        #     attempts += 1
        # except Exception as e:
        #     print(f"Code generation failed: {e}")
        #     failure_count += 1
        #     attempts += 1

if __name__ == "__main__":
    success_count = 0
    failure_count = 0
    op_args_generator = RandomNCHW()#conv_op.randomize_params()   #1改这个，随机生成参数的
    codegen_test = Codegen()# 2,全流程都一样，c，cuda，ir
    # 调用函数
    save_code(codegen_test,"rms_norm", rms_norm,op_args_generator=op_args_generator)  #3. 调用函数
    print(f"成功个数：{success_count}")
    print(f"失败个数：{failure_count}")

