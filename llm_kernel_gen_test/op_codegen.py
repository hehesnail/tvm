import tvm
import tvm.testing
from tvm import te
from tvm import topi
import numpy as np
import json
import os

filename = "data.json"
topi_operators_1d = []
topi_operators_2d = []
topi_operators_2s = []
topi_operators_3d = []
n = 128
A_1d = te.placeholder((n,), name = "A_1d")
B_1d = te.placeholder((n,), name = "B_1d")

A_2d = te.placeholder((n,n), name = "A_2d")


A_3d = te.placeholder((n,1,n), name = "A_3d")



def file_init(filename):
    if os.path.exists(filename):
        os.remove(filename)
        print(f"file {filename} reload")
def create_json_file(kernel_name,c_code, cuda_code, filename):
    data = {kernel_name + '_c': c_code, kernel_name + '_cuda': cuda_code}
    # with open(filename, 'w') as file:
    #     file.write(json.dumps(data)+'\n')
    
    with open(filename, 'a') as file:
        file.seek(0,2)
        file.write(json.dumps(data)+'\n')


def cuda_codegen_elemwise(kernel_topi,tensors):
    with tvm.target.Target("cuda"):
        sst = topi.cuda.schedule_elemwise(kernel_topi)
        rt_mod = tvm.build(sst, tensors)
        cuda_code = rt_mod.imported_modules[0].get_source()
    return cuda_code


def c_codegen(op,tensors):
    ts = te.create_schedule(op.op)
    c_code = tvm.build(ts, tensors, 'c').imported_modules[0].get_source()
    return c_code



def add_ops():
    # 添加topi算子到列表中
    topi_operators_1d.append([topi.cosh(A_1d),"cosh"])
    topi_operators_1d.append([topi.cos(A_1d),"cos"])
    topi_operators_1d.append([topi.acos(A_1d),"acos"])
    topi_operators_1d.append([topi.asin(A_1d),"asin"])
    topi_operators_1d.append([topi.asinh(A_1d),"asinh"])
    topi_operators_1d.append([topi.atan(A_1d),"atan"])
    topi_operators_1d.append([topi.atanh(A_1d),"atanh"])
    topi_operators_1d.append([topi.ceil(A_1d),"ceil"])
    topi_operators_1d.append([topi.clip(A_1d,a_min=0,a_max=100),"clip"])
    topi_operators_1d.append([topi.const_vector(np.ones((n)),"const_vector_np_ones"),"const_vector_np_ones"])
    topi_operators_1d.append([topi.const_vector(np.zeros((n)),"const_vector_np_zeros"),"const_vector_np_zeros"])
    topi_operators_1d.append([topi.const_vector(np.full((n), 7) ,"const_vector_np_full"),"const_vector_np_full"])
    topi_operators_1d.append([topi.const_vector(np.arange(0,10,2) ,"const_vector_np_arange"),"const_vector_np_arange"])    
    topi_operators_1d.append([topi.erf(A_1d),"erf"])    
    topi_operators_1d.append([topi.exp(A_1d),"exp"])
    topi_operators_1d.append([topi.fast_erf(A_1d),"fast_erf"])
    topi_operators_1d.append([topi.fast_exp(A_1d),"fast_exp"])
    topi_operators_1d.append([topi.fast_tanh(A_1d),"fast_tanh"])
    topi_operators_1d.append([topi.fixed_point_multiply(A_1d,2,2),"fixed_point_multiply"])
    topi_operators_1d.append([topi.flip(A_1d),"flip"])
    topi_operators_1d.append([topi.floor(A_1d),"floor"])
    topi_operators_1d.append([topi.full_like(A_1d,fill_value=1.0),"full_like"])  
    topi_operators_1d.append([topi.isnan(A_1d),"isnan"])
    topi_operators_1d.append([topi.log(A_1d),"log"])
    topi_operators_1d.append([topi.log10(A_1d),"log10"])
    topi_operators_1d.append([topi.log2(A_1d),"log2"])
    topi_operators_1d.append([topi.max(A_1d,keepdims=True),"max"])
    topi_operators_1d.append([topi.min(A_1d,keepdims=True),"min"])
    topi_operators_1d.append([topi.negative(A_1d),"negative"])
    topi_operators_1d.append([topi.prod(A_1d,axis=0,keepdims=True),"prod"])    
    topi_operators_1d.append([topi.reinterpret(A_1d,dtype='float32'),"reinterpret"])
    topi_operators_1d.append([topi.repeat(A_1d,repeats=1,axis=0),"repeat"])    
    topi_operators_1d.append([topi.reshape(A_1d,newshape=(2,n//2)),"reshape_1"])
    topi_operators_1d.append([topi.reshape(A_1d,newshape=(4,n//4)),"reshape_2"])
    topi_operators_1d.append([topi.round(A_1d),"round"])
    topi_operators_1d.append([topi.rsqrt(A_1d),"rsqrt"])
    topi_operators_1d.append([topi.shape(A_1d,dtype='int32'),"shape"])
    topi_operators_1d.append([topi.sigmoid(A_1d),"sigmoid"])
    topi_operators_1d.append([topi.sign(A_1d),"sign"])
    topi_operators_1d.append([topi.sin(A_1d),"sin"])
    topi_operators_1d.append([topi.sinh(A_1d),"sinh"])
    topi_operators_1d.append([topi.sqrt(A_1d),"sqrt"])
    topi_operators_1d.append([topi.tan(A_1d),"tan"])
    topi_operators_1d.append([topi.tanh(A_1d),"tanh"])
    topi_operators_1d.append([topi.tile(A_1d,reps=(1,2,3)),"tile"]) 

def add_ops_2d():
    topi_operators_2d.append([topi.divide(A_1d,B_1d),"divide"])
    topi_operators_2d.append([topi.equal(A_1d,B_1d),"equal"])
    topi_operators_2d.append([topi.concatenate((A_1d,B_1d),axis=0),"concatenate"])
    topi_operators_2d.append([topi.elemwise_sum([A_1d,A_1d]),"elemwise_sum"])    
    topi_operators_2d.append([topi.floor_divide(A_1d,B_1d),"floor_divide"])    
    topi_operators_2d.append([topi.floor_mod(A_1d,B_1d),"floor_mod"])
    topi_operators_2d.append([topi.greater(A_1d,A_1d),"greater"]) 
    topi_operators_2d.append([topi.greater_equal(A_1d,A_1d),"greater_equal"]) 
    topi_operators_2d.append([topi.less(A_1d,B_1d),"left_shift"])
    topi_operators_2d.append([topi.less_equal(A_1d,B_1d),"less_equal"])
    topi_operators_2d.append([topi.maximum(A_1d,B_1d),"maximum"])    
    topi_operators_2d.append([topi.minimum(A_1d,B_1d),"minimum"])
    topi_operators_2d.append([topi.mod(A_1d,B_1d),"mod"])
    topi_operators_2d.append([topi.multiply(A_1d,B_1d),"multiply"])
    topi_operators_2d.append([topi.not_equal(A_1d,B_1d),"not_equal"])    
    topi_operators_2d.append([topi.power(A_1d,B_1d),"power"])
    topi_operators_2d.append([topi.subtract(A_1d,B_1d),"stack"])
    topi_operators_2d.append([topi.tensordot(A_1d,B_1d,axes=0),"tensordot"])
def add_ops_2s():
    topi_operators_2s.append([topi.expand_dims(A_2d,0),"expand_dims"])    
    topi_operators_2s.append([topi.identity(A_2d),"identity"])
    topi_operators_2s.append([topi.repeat(A_2d,repeats=1,axis=0),"repeat_2d"])    
    topi_operators_2s.append([topi.reshape(A_2d,newshape=(128,2,64)),"reshape_1s"])
    topi_operators_2s.append([topi.reshape(A_2d,newshape=(n,n//2,2)),"reshape_1s"])
    topi_operators_2s.append([topi.reshape(A_2d,newshape=(n//2,2,n//2,2)),"reshape_2s"])
    topi_operators_2s.append([topi.reshape(A_2d,newshape=(n//4,4,n//2,2)),"reshape_3s"])
    topi_operators_2s.append([topi.reshape(A_2d,newshape=(n//4,4,n)),"reshape_4s"])
    topi_operators_2s.append([topi.reshape(A_2d,newshape=(n//8,8,n)),"reshape_5s"])
    topi_operators_2s.append([topi.reshape(A_2d,newshape=(n//4,4,n//8,8)),"reshape_4s"])
    topi_operators_2s.append([topi.shape(A_2d,dtype='int32'),"shape"])
    topi_operators_2s.append([topi.shape(A_2d,dtype='int32'),"shape"])
    topi_operators_2s.append([topi.tan(A_2d),"tan"])
    topi_operators_2s.append([topi.tanh(A_2d),"tanh"])
def add_ops_3d():
    topi_operators_3d.append([topi.squeeze(A_3d),"squeeze"])
    pass 
# open the functions
# add_ops()
# add_ops_2d()
# add_ops_2s()
# add_ops_3d()
file_init(filename)
success_count = 0
failure_count = 0

test_list = []
def test_1d():
    global test_list
    global A_1d
    global success_count

    for op, op_name in test_list:
        c_code = c_codegen(op, [A_1d])
        cuda_code = cuda_codegen_elemwise(op, [A_1d])
        create_json_file(op_name, c_code, cuda_code, filename)
        print(c_code)
        success_count += 1

def test_2d():
    global test_list
    global A_1d
    global B_1d
    global success_count
    
    for op, op_name in test_list:
        c_code = c_codegen(op, [A_1d,B_1d])
        cuda_code = cuda_codegen_elemwise(op, [A_1d,B_1d])
        create_json_file(op_name, c_code, cuda_code, filename)
        print(c_code)
        success_count += 1


def test_2s():
    global test_list
    global A_2d
    global B_1d
    global success_count
    
    for op, op_name in test_list:
        c_code = c_codegen(op, [A_2d])
        cuda_code = cuda_codegen_elemwise(op, [A_2d])
        create_json_file(op_name, c_code, cuda_code, filename)
        print(c_code)
        print(cuda_code)
        success_count += 1

def test_3d():
    global test_list
    global A_3d
    global success_count
    
    for op, op_name in test_list:
        c_code = c_codegen(op, [A_3d])
        cuda_code = cuda_codegen_elemwise(op, [A_3d])
        create_json_file(op_name, c_code, cuda_code, filename)
        print(c_code)
        success_count += 1
def test():
    pass
    # test_list.append([topi.trunc(A_2d),"trunc"])
    test_list.append([topi.tensordot(A_1d,B_1d,axes=0),"tensordot"])
    test_2d()
    # test_2s()
    # test_3d()
    
test()



def codegen(ops_list,args):
    global success_count
    global failure_count
    for op,op_name in ops_list:
        try:
            c_code = c_codegen(op, args)
            cuda_code = cuda_codegen_elemwise(op, args)
            create_json_file(op_name, c_code, cuda_code, filename)
            success_count += 1
        except Exception as e:
            failure_count += 1
            print(f"操作 {op_name} 失败: {str(e)}")


codegen(topi_operators_1d,[A_1d])
codegen(topi_operators_2d,[A_1d,B_1d])
codegen(topi_operators_2s,[A_2d])
codegen(topi_operators_3d,[A_3d])



print(f"成功个数：{success_count}")
print(f"失败个数：{failure_count}")












# for op, op_name in topi_operators_1d:
#     try:
#         c_code = c_codegen(op, [A_1d])
#         cuda_code = cuda_codegen_elemwise(op, [A_1d])
#         create_json_file(op_name, c_code, cuda_code, filename)
#         success_count += 1
#     except Exception as e:
#         failure_count += 1
#         print(f"操作 {op_name} 失败: {str(e)}")



# for op, op_name in topi_operators_2d:
#     try:
#         c_code = c_codegen(op, [A_1d,B_1d])
#         cuda_code = cuda_codegen_elemwise(op, [A_1d,B_1d])
#         create_json_file(op_name, c_code, cuda_code, filename)
#         success_count += 1
#     except Exception as e:
#         failure_count += 1
#         print(f"操作 {op_name} 失败: {str(e)}")


# for op, op_name in topi_operators_2s:
#     try:
#         c_code = c_codegen(op, [A_2d])
#         cuda_code = cuda_codegen_elemwise(op, [A_2d])
#         create_json_file(op_name, c_code, cuda_code, filename)
#         success_count += 1
#     except Exception as e:
#         failure_count += 1
#         print(f"操作 {op_name} 失败: {str(e)}")

# 创建JSON文件

# kernel_name = "cosh"
# create_json_file(kernel_name,"aa" ,"bb", filename)
