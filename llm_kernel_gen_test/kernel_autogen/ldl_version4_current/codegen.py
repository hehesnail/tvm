import numpy as np
import tvm
from tvm import te, auto_scheduler,topi
from tvm.autotvm import measure, task, tuner

import os
import shutil
import json

topi_operators_1d = []
ops_json_data = []
idx = 0
A = None
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
    # 添加topi算子到列表中
    topi_operators_1d.append(topi.fast_tanh)
    topi_operators_1d.append(topi.fixed_point_multiply)
    topi_operators_1d.append(topi.flip)
    topi_operators_1d.append(topi.floor)
    topi_operators_1d.append(topi.full_like)
    topi_operators_1d.append(topi.isnan)
    topi_operators_1d.append(topi.max)
    topi_operators_1d.append(topi.min)
    topi_operators_1d.append(topi.prod)
    topi_operators_1d.append(topi.reinterpret)
    topi_operators_1d.append(topi.repeat)
    topi_operators_1d.append(topi.reshape)
    topi_operators_1d.append(topi.shape)
    topi_operators_1d.append(topi.tile)


    topi_operators_1d.append(topi.erf)
    topi_operators_1d.append(topi.exp)
    topi_operators_1d.append(topi.fast_erf)
    topi_operators_1d.append(topi.fast_exp)
    topi_operators_1d.append(topi.negative)
    topi_operators_1d.append(topi.log)
    topi_operators_1d.append(topi.log10)
    topi_operators_1d.append(topi.log2)
    topi_operators_1d.append(topi.rsqrt)
    topi_operators_1d.append(topi.round)
    topi_operators_1d.append(topi.sigmoid)
    topi_operators_1d.append(topi.sign)
    topi_operators_1d.append(topi.sin)
    topi_operators_1d.append(topi.sinh)
    topi_operators_1d.append(topi.sqrt)
    topi_operators_1d.append(topi.tan)
    topi_operators_1d.append(topi.tanh)
    topi_operators_1d.append(topi.cosh)
    topi_operators_1d.append(topi.cos)
    topi_operators_1d.append(topi.acos)
    topi_operators_1d.append(topi.asin)
    topi_operators_1d.append(topi.asinh)
    topi_operators_1d.append(topi.atan)
    topi_operators_1d.append(topi.atanh)
    topi_operators_1d.append(topi.ceil)
    topi_operators_1d.append(topi.clip)
    

def topi_operation( dtype,func):
    global A 
    out = func(A)
    return [A, out]

@auto_scheduler.register_workload
def topi_ops(dtype,*args ,func=""):

    return topi_operation( dtype,topi_operators_1d[idx])
    
def c_codegen(search_times = 50,shape = ""):
    target_c = tvm.target.Target("llvm")
    task_c = tvm.auto_scheduler.SearchTask(func=topi_ops, args=("float32",*(int(x) for x in shape) ), target=target_c)

    reload_file(log_file)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=search_times,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    # Run auto-tuning (search)
    task_c.tune(tune_option)

    # Apply the best schedule
    sch, args = task_c.apply_best(log_file)
    c_module = tvm.build(sch, args,"c")
    c_code   = c_module.get_source()
    lowered_ir = tvm.lower(sch, args, simple_mode=True)

    return c_code,str(lowered_ir)

def cuda_codegen(search_times = 100,shape = ""):
    target_cuda = tvm.target.Target("cuda")
    task_cuda = tvm.auto_scheduler.SearchTask(func=topi_ops, args=("float32",*(int(x) for x in shape) ), target=target_cuda)
    # print("task_cuda DAG:")
    # print(task_cuda.compute_dag)

    reload_file(log_file)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=search_times,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
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

def create_json_file(kernel_name,c_code, cuda_code, filename):
    data = {kernel_name + '_c': c_code, kernel_name + '_cuda': cuda_code}
    # with open(filename, 'w') as file:
    #     file.write(json.dumps(data)+'\n')

def json_file(filename):
    global ops_json_data
    with open(filename, 'a') as file:
        # for item in ops_json_data:
        file.seek(0,2)
        file.write(json.dumps(ops_json_data)+'\n')
    # all_ops_json = json.dumps(ops_json_data, indent=4)

N = L = M = 64 #TODO hannibal for random shape
add_ops()
max_1d_len = len(topi_operators_1d)


c_path    = "generate/c/"
cuda_path = "generate/cuda/"
create_dir(cuda_path)
create_dir(c_path)


success_count = 0
failure_count = 0
filename = "data.json"
failed_case = []

for i in range(max_1d_len):
    name = topi_operators_1d[idx].__name__
    for i in range(1,5):
        for j in range(5):
            random_shape = tuple(np.random.randint(8, 40, size=i))
            A = te.placeholder(random_shape, name="tarray",dtype="float32")
            print(random_shape)
            print(A)
            if i >=2 :
                op_name = name + str(random_shape).replace(",", "_").replace(" ", "_")
            elif i == 1:
                op_name = name + str(random_shape).replace(",", "").replace(" ", "_")
            try:
                print("c_codegen")
                c_code,ir_code = c_codegen(shape=random_shape)
                print("cuda_codegen")
                cuda_code = cuda_codegen(shape=random_shape)
                op_data = {
                'op_name': op_name,
                'c_code': c_code,
                'cuda_code': cuda_code,
                'ir_code': ir_code
                }
                ops_json_data.append(op_data)

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

json_file(filename)