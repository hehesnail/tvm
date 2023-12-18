import tvm
from tvm import relay
import tvm.testing
from tvm import te
from tvm import topi
import numpy as np
import json
import os

filename = "data.json"
topi_operators_random = []
topi_operators_1d = []
topi_operators_2d = []
topi_operators_2s = []
topi_operators_3d = []
ops_json_data = []
n = 128
A_1d = te.placeholder((n,), name = "A_1d")
B_1d = te.placeholder((n,), name = "B_1d")
A_2d = te.placeholder((n,n), name = "A_2d")
A_3d = te.placeholder((n,1,n), name = "A_3d")
batch_size = 2
M = 4
N = 3
K = 5
dtype = "float32"
tensor_a = te.placeholder((batch_size, M, K), dtype=dtype, name="tensor_a")
tensor_b = te.placeholder((batch_size, N, K), dtype=dtype, name="tensor_b")
tensor_a = te.placeholder((batch_size, M, K), dtype=dtype, name="tensor_a")
tensor_b = te.placeholder((batch_size, N, K), dtype=dtype, name="tensor_b")
out_dtype = dtype
transpose_b = True
transpose_a = False
oshape = (batch_size, M, N)



def file_init(filename):
    if os.path.exists(filename):
        os.remove(filename)
        print(f"file {filename} reload")


def json_file(filename):
    global ops_json_data
    with open(filename, 'a') as file:
        # for item in ops_json_data:
        file.seek(0,2)
        file.write(json.dumps(ops_json_data)+'\n')
    # all_ops_json = json.dumps(ops_json_data, indent=4)
    
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
        cuda_code = rt_mod.get_source()
    return cuda_code


def c_codegen(op,tensors):
    ts = te.create_schedule(op.op)
    rt_mod = tvm.build(ts, tensors, 'c')
    c_code = rt_mod.get_source()
    return c_code

def ir_codegen(op,tensors):
    ts = te.create_schedule(op.op)
    lowered_ir = tvm.lower(ts, tensors, simple_mode=True)
    return str(lowered_ir)
    


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

def add_random(tarray,newshape=None):
    
    topi_operators_random.append([topi.cosh(tarray),"cosh"])
    topi_operators_random.append([topi.cos(tarray),"cos"])
    topi_operators_random.append([topi.acos(tarray),"acos"])
    topi_operators_random.append([topi.asin(tarray),"asin"])
    topi_operators_random.append([topi.asinh(tarray),"asinh"])
    topi_operators_random.append([topi.atan(tarray),"atan"])
    topi_operators_random.append([topi.atanh(tarray),"atanh"])
    topi_operators_random.append([topi.ceil(tarray),"ceil"])
    topi_operators_random.append([topi.clip(tarray,a_min=0,a_max=100),"clip"])    
    topi_operators_random.append([topi.erf(tarray),"erf"])    
    topi_operators_random.append([topi.exp(tarray),"exp"])
    topi_operators_random.append([topi.fast_erf(tarray),"fast_erf"])
    topi_operators_random.append([topi.fast_exp(tarray),"fast_exp"])
    topi_operators_random.append([topi.fast_tanh(tarray),"fast_tanh"])
    topi_operators_random.append([topi.fixed_point_multiply(tarray,2,2),"fixed_point_multiply"])
    topi_operators_random.append([topi.flip(tarray),"flip"])
    topi_operators_random.append([topi.floor(tarray),"floor"])
    topi_operators_random.append([topi.full_like(tarray,fill_value=1.0),"full_like"])  
    topi_operators_random.append([topi.isnan(tarray),"isnan"])
    topi_operators_random.append([topi.log(tarray),"log"])
    topi_operators_random.append([topi.log10(tarray),"log10"])
    topi_operators_random.append([topi.log2(tarray),"log2"])
    topi_operators_random.append([topi.max(tarray,keepdims=True),"max"])
    topi_operators_random.append([topi.min(tarray,keepdims=True),"min"])
    topi_operators_random.append([topi.negative(tarray),"negative"])
    topi_operators_random.append([topi.prod(tarray,axis=0,keepdims=True),"prod"])    
    topi_operators_random.append([topi.reinterpret(tarray,dtype='float32'),"reinterpret"])
    topi_operators_random.append([topi.repeat(tarray,repeats=1,axis=0),"repeat"])    
    topi_operators_random.append([topi.reshape(tarray,newshape=(2,n//2)),"reshape_1"])
    topi_operators_random.append([topi.reshape(tarray,newshape=(4,n//4)),"reshape_2"])
    topi_operators_random.append([topi.round(tarray),"round"])
    topi_operators_random.append([topi.rsqrt(tarray),"rsqrt"])
    topi_operators_random.append([topi.shape(tarray,dtype='int32'),"shape"])
    topi_operators_random.append([topi.sigmoid(tarray),"sigmoid"])
    topi_operators_random.append([topi.sign(tarray),"sign"])
    topi_operators_random.append([topi.sin(tarray),"sin"])
    topi_operators_random.append([topi.sinh(tarray),"sinh"])
    topi_operators_random.append([topi.sqrt(tarray),"sqrt"])
    topi_operators_random.append([topi.tan(tarray),"tan"])
    topi_operators_random.append([topi.tanh(tarray),"tanh"])
    topi_operators_random.append([topi.tile(tarray,reps=(1,2,3)),"tile"]) 


    topi_operators_random.append([topi.expand_dims(tarray,0),"expand_dims"])    
    topi_operators_random.append([topi.identity(tarray),"identity"])
    if newshape != None:
        topi_operators_random.append([topi.reshape(tarray,newshape),"reshape"])


    topi_operators_random.append([topi.const_vector(np.ones((n)),"const_vector_np_ones"),"const_vector_np_ones"])
    topi_operators_random.append([topi.const_vector(np.zeros((n)),"const_vector_np_zeros"),"const_vector_np_zeros"])
    topi_operators_random.append([topi.const_vector(np.full((n), 7) ,"const_vector_np_full"),"const_vector_np_full"])
    topi_operators_random.append([topi.const_vector(np.arange(0,10,2) ,"const_vector_np_arange"),"const_vector_np_arange"])

def add_random1(ndarray,newshape=None):
    topi_operators_random.append([topi.divide(ndarray,ndarray),"divide"])
    topi_operators_random.append([topi.equal(ndarray,ndarray),"equal"])
    topi_operators_random.append([topi.concatenate((ndarray,ndarray),axis=0),"concatenate"])
    topi_operators_random.append([topi.elemwise_sum([ndarray,ndarray]),"elemwise_sum"])    
    topi_operators_random.append([topi.floor_divide(ndarray,ndarray),"floor_divide"])    
    topi_operators_random.append([topi.floor_mod(ndarray,ndarray),"floor_mod"])
    topi_operators_random.append([topi.greater(ndarray,ndarray),"greater"]) 
    topi_operators_random.append([topi.greater_equal(ndarray,ndarray),"greater_equal"]) 
    topi_operators_random.append([topi.less(ndarray,ndarray),"left_shift"])
    topi_operators_random.append([topi.less_equal(ndarray,ndarray),"less_equal"])
    topi_operators_random.append([topi.maximum(ndarray,ndarray),"maximum"])    
    topi_operators_random.append([topi.minimum(ndarray,ndarray),"minimum"])
    topi_operators_random.append([topi.mod(ndarray,ndarray),"mod"])
    topi_operators_random.append([topi.multiply(ndarray,ndarray),"multiply"])
    topi_operators_random.append([topi.not_equal(ndarray,ndarray),"not_equal"])    
    topi_operators_random.append([topi.power(ndarray,ndarray),"power"])
    topi_operators_random.append([topi.subtract(ndarray,ndarray),"stack"])
    topi_operators_random.append([topi.tensordot(ndarray,ndarray,axes=0),"tensordot"])
    pass



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

    # blow are nn ops



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

    # blow are nn ops
    topi_operators_2d.append([topi.nn.add(A_1d,B_1d),"trunc"])

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

    # blow are nn ops


def add_ops_3d():
    topi_operators_3d.append([topi.squeeze(A_3d),"squeeze"])
    pass 
# open the functions

list_special_ops = []
def add_special_ops():
    list_special_ops.append([
        [[topi.nn.batch_matmul(tensor_a, tensor_b, oshape, out_dtype, transpose_a, transpose_b),"test"]],
        [tensor_a, tensor_b,topi.nn.batch_matmul(tensor_a, tensor_b, oshape, out_dtype, transpose_a, transpose_b)]
        ])
    pass
    # list_special_ops


def codegen(ops_list,args):
    global success_count
    global failure_count
    global ops_json_data

    for op,op_name in ops_list:
        print(op)
        print(len(ops_list))
        
        try:
            c_code = c_codegen(op, args)
            cuda_code = cuda_codegen_elemwise(op, args)
            ir_code  = ir_codegen(op,args)
            op_data = {
                'op_name': op_name,
                'c_code': c_code,
                'cuda_code': cuda_code,
                'ir_code': ir_code
            }
            ops_json_data.append(op_data)
            success_count += 1
        except Exception as e:
            failure_count += 1
            print(f"操作 {op_name} 失败: {str(e)}")   # op.name

def codegen_for_special_ops():
    for lis_op,op_args in list_special_ops:
        codegen(lis_op,op_args)


def verify_adaptive_pool(dshape, out_size, pool_type, layout="NCHW", dtype="float32"):
    """verify function of adaptive_pool"""
    
    data = te.placeholder(dshape, name="data", dtype=dtype)
    if len(out_size) == 2:
        out =  relay.nn.adaptive_avg_pool2d(data, out_size, pool_type, layout) if pool_type == "avg" else relay.nn.adaptive_max_pool2d(data, out_size, pool_type, layout) #topi.nn.adaptive_pool(data, out_size, pool_type, layout)
    else:
        assert len(out_size) == 3
        out = topi.nn.adaptive_pool3d(data, out_size, pool_type, layout)
    tensors = []
    tensors.append(data)
    ts = topi.cuda.schedule_adaptive_pool(out.op)
    cuda_code = tvm.build(ts,[data,out ], 'cuda').imported_modules[0].get_source()
    print(cuda_code)




for i in range(1, 7):
    for j in range(1,5):
        random_shape = tuple(np.random.randint(1, 11, size=i),)
        new_shape = tuple(np.random.permutation(random_shape))
        tarray = te.placeholder(random_shape, name="tarray")
        print(tarray)
        # add_random(tarray,new_shape)
        add_random(tarray,new_shape)
        codegen(topi_operators_random,[tarray])
        topi_operators_random=[]
# 创建 TVM 的输入占位符，并将随机形状赋值给它

json_file(filename)

print(f"成功个数：{success_count}")
print(f"失败个数：{failure_count}")



