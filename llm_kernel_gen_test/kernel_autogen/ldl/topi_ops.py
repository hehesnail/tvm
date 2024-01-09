import numpy as np
import tvm
from tvm import te, auto_scheduler,topi
from tvm.autotvm import measure, task, tuner

import os
import io
import shutil
import json

import signal
import time


# A tensor,B tensor
A = None  
B = None
idx = 0

bool_tensors = ['all']

idx_2d = 0 
success_count = 0
failure_count = 0
failed_case = []
ops_json_data = []
topi_operators_1d = []

topi_operators_2d = []
filename = "data.json"
log_file = "codegen.json"

def reload_file(log_file):
    if os.path.exists(log_file):
        # 如果文件存在，则清空文件内容
        with open(log_file, "w") as f:
            pass
    else:
        # 如果文件不存在，则创建空文件
        with open(log_file, "x"):
            pass

def add_ops():
    topi_operators_1d.append(topi.abs)
    topi_operators_1d.append(topi.acos)
    topi_operators_1d.append(topi.acosh)
    topi_operators_1d.append(topi.asin)
    topi_operators_1d.append(topi.asinh)
    topi_operators_1d.append(topi.atan)
    topi_operators_1d.append(topi.atanh)
    topi_operators_1d.append(topi.ceil)
    topi_operators_1d.append(topi.ceil_log2)
    topi_operators_1d.append(topi.cos)
    topi_operators_1d.append(topi.cosh)
    topi_operators_1d.append(topi.erf)
    topi_operators_1d.append(topi.exp)
    topi_operators_1d.append(topi.fast_erf)
    topi_operators_1d.append(topi.fast_exp)
    topi_operators_1d.append(topi.fast_tanh)
    topi_operators_1d.append(topi.floor)
    topi_operators_1d.append(topi.identity)
    topi_operators_1d.append(topi.invert_permutation)
    topi_operators_1d.append(topi.isfinite)
    topi_operators_1d.append(topi.isinf)
    topi_operators_1d.append(topi.isnan)
    topi_operators_1d.append(topi.log)
    topi_operators_1d.append(topi.log10)
    topi_operators_1d.append(topi.log2)
    topi_operators_1d.append(topi.logical_not)
    topi_operators_1d.append(topi.max)
    topi_operators_1d.append(topi.min)
    topi_operators_1d.append(topi.ndarray_size)
    topi_operators_1d.append(topi.negative)
    topi_operators_1d.append(topi.round)
    topi_operators_1d.append(topi.rsqrt)
    topi_operators_1d.append(topi.sigmoid)
    topi_operators_1d.append(topi.sign)
    topi_operators_1d.append(topi.sin)
    topi_operators_1d.append(topi.sinh)
    topi_operators_1d.append(topi.sqrt)
    topi_operators_1d.append(topi.tan)
    topi_operators_1d.append(topi.tanh)
    topi_operators_1d.append(topi.all)
    topi_operators_1d.append(topi.any)
    topi_operators_1d.append(topi.flip)


def add_2dops():
    topi_operators_2d.append(topi.add)
    topi_operators_2d.append(topi.divide)
    topi_operators_2d.append(topi.equal)
    topi_operators_2d.append(topi.floor_divide)
    topi_operators_2d.append(topi.floor_mod)
    topi_operators_2d.append(topi.greater)
    topi_operators_2d.append(topi.greater_equal)
    topi_operators_2d.append(topi.less)
    topi_operators_2d.append(topi.less_equal)
    topi_operators_2d.append(topi.logical_and)
    topi_operators_2d.append(topi.logical_or)
    topi_operators_2d.append(topi.logical_xor)
    topi_operators_2d.append(topi.maximum)
    topi_operators_2d.append(topi.minimum)
    topi_operators_2d.append(topi.mod)
    topi_operators_2d.append(topi.multiply)
    topi_operators_2d.append(topi.not_equal)
    topi_operators_2d.append(topi.power)
    topi_operators_2d.append(topi.right_shift)
    topi_operators_2d.append(topi.subtract)

def topi_operation( dtype,func):
    global A 
    out = func(A)
    return [A, out]

def topi_2d_operation( dtype,func):
    global A 
    global B
    out = func(A,B)
    return [A,B, out]

@auto_scheduler.register_workload
def topi_ops(dtype,*args ,func=""):
    return topi_operation( dtype,topi_operators_1d[idx])

@auto_scheduler.register_workload
def topi_2d_ops(dtype,*args ,func=""):
    return topi_2d_operation( dtype,topi_operators_2d[idx_2d])


class DummyFile(io.StringIO):
    def write(self, x):
        pass


def c_codegen(search_times = 2,shape = "",verbose_key=0,nd_tensors_numbers = 1):
    target_c = tvm.target.Target("llvm")
    if nd_tensors_numbers == 1:
        task_c = tvm.auto_scheduler.SearchTask(func=topi_ops, args=("float32",*(int(x) for x in shape) ), target=target_c)
    if nd_tensors_numbers == 2:
        task_c = tvm.auto_scheduler.SearchTask(func=topi_2d_ops, args=("float32",*(int(x) for x in shape) ), target=target_c)

    reload_file(log_file)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=search_times,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=verbose_key,
    )
    # Run auto-tuning (search)
    task_c.tune(tune_option)

    # Apply the best schedule
    sch, args = task_c.apply_best(log_file)
    c_module = tvm.build(sch, args,"c")
    c_code   = c_module.get_source()
    lowered_ir = tvm.lower(sch, args, simple_mode=True)

    return c_code,str(lowered_ir)

def cuda_codegen(search_times = 2,shape = "",verbose_key=0,nd_tensors_numbers = 1):
    target_cuda = tvm.target.Target("cuda")
    if nd_tensors_numbers == 1:
        task_cuda = tvm.auto_scheduler.SearchTask(func=topi_ops, args=("float32",*(int(x) for x in shape) ), target=target_cuda)
    if nd_tensors_numbers == 2:
        task_cuda = tvm.auto_scheduler.SearchTask(func=topi_2d_ops, args=("float32",*(int(x) for x in shape) ), target=target_cuda)

    reload_file(log_file)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=search_times,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=verbose_key,
    )

    # Run auto-tuning (search)
    task_cuda.tune(tune_option)
    
    # Apply the best schedule
    sch, args = task_cuda.apply_best(log_file)

    cuda_module = tvm.build(sch, args,"cuda")
    cuda_code   = cuda_module.imported_modules[0].get_source()
    return cuda_code

def save_string_to_file(string, file_path):
    with open(file_path, 'w') as file:
        file.write(string)

def create_dir(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    else:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

def json_file(filename):
    global ops_json_data
    with open(filename, 'a') as file:
        file.seek(0,2)
        file.write(json.dumps(ops_json_data)+'\n')



c_path    = "generate/c/"
cuda_path = "generate/cuda/"
create_dir(cuda_path)
create_dir(c_path)


add_ops()
add_2dops()
max_1d_len = len(topi_operators_1d)
max_2d_len = len(topi_operators_2d)



def handle_timeout(signum, frame):
    raise KeyboardInterrupt('Timeout occurred.')  # 强制抛出 KeyboardInterrupt 异常

for i in range(max_2d_len):
    name = topi_operators_2d[idx_2d].__name__
    for i in range(1, 7):
        for j in range(5):
            random_shape = tuple(np.random.randint(4, 30, size=i))
            A = te.placeholder(random_shape, name="tarray", dtype="float32")
            B = te.placeholder(random_shape, name="tarray", dtype="float32")
            op_name = name
            try:
                print("c_codegen")
                signal.signal(signal.SIGALRM, handle_timeout)  # 设置信号处理函数
                signal.alarm(60*5)  # 设置定时器
                c_code, ir_code = c_codegen(shape=random_shape, verbose_key=False, search_times=4, nd_tensors_numbers=2)
                signal.alarm(0)  # 取消定时器

                print("cuda_codegen")
                signal.signal(signal.SIGALRM, handle_timeout)  # 设置信号处理函数
                signal.alarm(60*5)  # 设置定时器
                cuda_code = cuda_codegen(shape=random_shape, verbose_key=False, search_times=4, nd_tensors_numbers=2)
                signal.alarm(0)  # 取消定时器

                shape_str = '_'.join(map(str, random_shape))
                op_data = {
                    'op_name': op_name,
                    'c_code': c_code,
                    'cuda_code': cuda_code,
                    'ir_code': ir_code,
                    'shape': shape_str
                }
                json_str = json.dumps(op_data)

                # Write the JSON string to file
                with open('topi_data_2d_nobroadcast.json', 'a') as f:
                    f.write(json_str + '\n')
                # ops_json_data.append(op_data)

                save_string_to_file(c_code, c_path + op_name + ".c")
                save_string_to_file(cuda_code, cuda_path + op_name + ".cu")
                success_count += 1

            except KeyboardInterrupt as e:
                print(f"Timeout occurred while processing {name}. Skipping to next iteration.")
                continue

            except Exception as e:
                print(f"An error occurred while processing {name}: {e}")
                failure_count += 1
                failed_case.append(name + str(random_shape))
                continue
    idx_2d += 1


for i in range(max_1d_len):
    name = topi_operators_1d[idx].__name__
    for i in range(1,7):
        for j in range(5):
            print("success_count:"+str(success_count))
            print("failure_count:"+str(failure_count))
            random_shape = tuple(np.random.randint(4, 10, size=i))
            if name in bool_tensors:
                A = te.placeholder(random_shape, name="tarray",dtype="bool")
            else:
                A = te.placeholder(random_shape, name="tarray",dtype="float32")
            # op_name = name + str(random_shape).replace(",", "_").replace(" ", "_")
            op_name = name
            try:
                print("c_codegen")
                signal.signal(signal.SIGALRM, handle_timeout)  # 设置信号处理函数
                signal.alarm(60*10)  # 设置定时器
                c_code,ir_code = c_codegen(shape=random_shape,verbose_key=False,search_times=10)
                signal.alarm(0)  # 取消定时器

                print("cuda_codegen")
                signal.signal(signal.SIGALRM, handle_timeout)  # 设置信号处理函数
                signal.alarm(60*10)  # 设置定时器
                cuda_code = cuda_codegen(shape=random_shape,verbose_key=False,search_times=10)
                signal.alarm(0)  # 取消定时器


                shape_str = '_'.join(map(str, random_shape))
                op_data = {
                    'op_name': op_name,
                    'c_code': c_code,
                    'cuda_code': cuda_code,
                    'ir_code': ir_code,
                    'shape': shape_str
                }
                json_str = json.dumps(op_data)
                
                # Write the JSON string to file
                with open('topi_data_1d_nobroadcast.json', 'a') as f:
                    f.write(json_str + '\n')
                # ops_json_data.append(op_data)

                save_string_to_file(c_code, c_path + op_name+ ".c")
                save_string_to_file(cuda_code, cuda_path + op_name+ ".cu")
                success_count += 1
                
            except Exception as e:
                print(f"An error occurred while processing {name}: {e}")
                failure_count += 1
                failed_case.append(name+str(random_shape))
                continue
    idx += 1
    

print(f"Successfully processed {success_count} operators")
print(f"Failed to process {failure_count} operators:")
for i in failed_case:
    print(i)

