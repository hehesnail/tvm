import os
import numpy as np
import tvm
import tvm.testing
from tvm import te, auto_scheduler, runtime, topi
from tvm.auto_scheduler import _ffi_api
from tvm.topi.utils import get_const_tuple
from tvm.topi.sparse.utils import random_bsr_matrix
import json
from nnop_random  import RandomConvOperator


@auto_scheduler.register_workload
def conv2d_layer(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, bias, conv]


class Codegen:
    def __init__(self, search_times=2, shape="", verbose_key=0, log_file="temp_data.json"):
        self.search_times = search_times
        self.shape = shape
        self.verbose_key = verbose_key
        self.log_file = log_file

    def reload_file(self, log_file):
        if os.path.exists(log_file):
            # 如果文件存在，则清空文件内容
            with open(log_file, "w") as f:
                pass
        else:
            # 如果文件不存在，则创建空文件
            with open(log_file, "x"):
                pass

    def c_codegen(self, topi_ops=None, op_args=None):
        if topi_ops is None:
            raise ValueError("topi_ops must not be None and needs to be provided when c_codegen.")
        if op_args is None:
            raise ValueError("op_args must not be None and needs to be provided when c_codegen.")

        target_c = tvm.target.Target("llvm")
        task_c = tvm.auto_scheduler.SearchTask(func=topi_ops, args=(*op_args, ), target=target_c)

        self.reload_file(self.log_file)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=self.search_times,
            measure_callbacks=[auto_scheduler.RecordToFile(self.log_file)],
            verbose=self.verbose_key,
        )

        # Run auto-tuning (search)
        task_c.tune(tune_option)

        # Apply the best schedule
        sch, args = task_c.apply_best(self.log_file)
        c_module = tvm.build(sch, args, "c")
        c_code = c_module.get_source()
        lowered_ir = tvm.lower(sch, args, simple_mode=True)

        return c_code, str(lowered_ir)

    def cuda_codegen(self, topi_ops=None, op_args=None):
        if topi_ops is None:
            raise ValueError("topi_ops must not be None and needs to be provided when cuda_codegen.")
        if op_args is None:
            raise ValueError("op_args must not be None and needs to be provided when cuda_codegen.")

        target_cuda = tvm.target.Target("cuda")
        task_cuda = tvm.auto_scheduler.SearchTask(func=topi_ops, args=(*op_args, ), target=target_cuda)

        self.reload_file(self.log_file)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=self.search_times,
            measure_callbacks=[auto_scheduler.RecordToFile(self.log_file)],
            verbose=self.verbose_key,
        )

        # Run auto-tuning (search)
        task_cuda.tune(tune_option)

        # Apply the best schedule
        sch, args = task_cuda.apply_best(self.log_file)

        cuda_module = tvm.build(sch, args, "cuda")
        cuda_code = cuda_module.imported_modules[0].get_source()
        return cuda_code

def write_json_to_file(op_name, c_code, cuda_code, ir_code,save_file='test_data.json'):
    op_data = {
        'op_name': op_name,
        'c_code': c_code,
        'cuda_code': cuda_code,
        'ir_code': ir_code
    }
    json_str = json.dumps(op_data)
    with open(save_file, 'a') as f:
        f.write(json_str + ',\n')




# this code just FYI, show how to use it to generate data
if __name__ == "__main__":
    conv_op = RandomConvOperator()#conv_op.randomize_params()
    op_args = conv_op.get_param_values()

    codegen_test = Codegen()
    c_code,ir_code = codegen_test.c_codegen(topi_ops=conv2d_layer,op_args=op_args)
    cuda_code = codegen_test.cuda_codegen(topi_ops=conv2d_layer,op_args=op_args)
    op_name="conv" + str(op_args).replace(",", "_").replace(" ", "_")
    write_json_to_file(op_name,c_code,cuda_code,ir_code)



