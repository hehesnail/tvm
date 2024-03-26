import os
import tvm
import re

from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_executor
import tvm.testing
from tvm import topi


def parse_filename(filename):
    """
    解析给定的文件名，提取算子名称和shape信息。
    
    参数:
    - filename: 字符串，待解析的文件名。
    
    返回:
    - operator: 字符串，算子名称。
    - shape: 元组，shape信息。
    """
    # 使用正则表达式匹配算子名称和shape信息
    match = re.match(r"(\w+)\(([\d_]+)\)\.ll$", filename)
    # match = re.match(r"(\w+)\(([\d_]+(?:__[\d_]+)*)\)\.ll$", filename)
    if not match:
        raise ValueError("文件名格式不正确")
    
    operator, shape_str = match.groups()
    # 将shape信息转换为元组形式

    # print(shape_str)
    shape = tuple(map(int, shape_str.split('__')))
    
    return operator, shape

# 示例
# filename = "acos(75__62__65).py"
# operator, shape = parse_filename(filename)



import tvm
from tvm import te
import numpy as np

def compute_with_operator(operator_name, input_data):
    """
    根据算子名称和输入数据执行计算。
    
    参数:
        operator_name (str): 算子名称。
        input_data (numpy.ndarray): 输入数据。
        
    返回:
        numpy.ndarray: 计算结果。
    """
    # 获取输入数据的shape
    shape = input_data.shape
    print(shape)
    # 使用TVM定义输入张量
    A = te.placeholder(shape, dtype="float32", name="A")
    # 根据算子名称选择相应的TVM算子

    if operator_name == "acos":
        B = te.compute(A.shape, lambda *i: te.acos(A(*i)), name="B")
    elif operator_name == "asin":
        B = te.compute(A.shape, lambda *i: te.asin(A(*i)), name="B")
    elif operator_name == "asinh":
        B = te.compute(A.shape, lambda *i: te.asinh(A(*i)), name="B")
    elif operator_name == "atan":
        B = te.compute(A.shape, lambda *i: te.atan(A(*i)), name="B")
    elif operator_name == "atanh":
        B = te.compute(A.shape, lambda *i: te.atanh(A(*i)), name="B")
    elif operator_name == "ceil":
        B = te.compute(A.shape, lambda *i: te.ceil(A(*i)), name="B")
    elif operator_name == "erf":
        B = te.compute(A.shape, lambda *i: te.erf(A(*i)), name="B")
    elif operator_name == "exp":
        B = te.compute(A.shape, lambda *i: te.exp(A(*i)), name="B")
    elif operator_name == "fast_erf":
        B = topi.fast_erf(A)
    elif operator_name == "fast_exp":
        B = topi.fast_exp(A)
    elif operator_name == "flip":
        B = topi.flip(A)
    elif operator_name == "floor":
        B = topi.floor(A)
    elif operator_name == "log2":
        B = topi.log2(A)
    elif operator_name == "log10":
        B = topi.log10(A)
    elif operator_name == "negative":
        B = topi.negative(A)
    elif operator_name == "round":
        B = topi.round(A)
    elif operator_name == "log":
        B = topi.log(A)
    elif operator_name == "sin":
        B = topi.sin(A)
    elif operator_name == "tan":
        B = topi.tan(A)
    elif operator_name == "tanh":
        B = topi.tanh(A)
    elif operator_name == "rsqrt":
        B = topi.rsqrt(A)
    elif operator_name == "sqrt":
        B = topi.sqrt(A)
    elif operator_name == "sigmoid":
        B = topi.sigmoid(A)
    elif operator_name == "sign":
        B = topi.sign(A)
    elif operator_name == "sinh":
        B = topi.sinh(A)
    elif operator_name == "fast_tanh":
        B = topi.fast_tanh(A)
    elif operator_name == "shape":
        B = topi.shape(A)
    # elif operator_name == "shape":
    #     B = topi.shape(A)
    # elif operator_name == "sum":
    #     B = topi.sum(A)
    elif operator_name == "max":
        B = topi.max(A,axis=1)
    elif operator_name == "adaptive_pool_avg":
        print("A:",A)
        B = topi.nn.adaptive_pool(A,output_size=(8,8),pool_type='avg')
        
    elif operator_name == "add":
        # A1 = te.placeholder(shape, dtype="float32", name="A1")
        # B = topi.nn.add(A,A1)
        # s = te.create_schedule(B.op)
        # fadd = tvm.build(s,[A,A1,B],"llvm")
        # ctx = tvm.cpu(0)
        # lhs_tvm = tvm.nd.array(input_data, ctx)
        # rhs_tvm = tvm.nd.array(input_data, ctx)
        # output_tvm = tvm.nd.array(np.zeros(input_data.shape), ctx)
        # print("------------------=-=-==============1")
        # fadd(lhs_tvm, rhs_tvm, output_tvm)
        # print("------------------=-=-==============2")
        # print(output_tvm.asnumpy())
        # return output_tvm.asnumpy()
        # B = topi.max(A)
        dtype = "float32"
        A = te.placeholder(shape, dtype=dtype, name='A')
        B = te.placeholder(shape, dtype=dtype, name='B')

        # 执行加法操作
        C = topi.nn.add(A, B)

        # 创建调度器
        s = te.create_schedule([C.op])

        # 生成代码
        mod = tvm.build(s, [A, B, C])

        # 创建TVM运行时环境
        ctx = tvm.cpu(0)
        a = tvm.nd.array(input_data, ctx)
        b = tvm.nd.array(input_data, ctx)
        c = tvm.nd.empty(shape, dtype=dtype)

        # 执行计算
        mod(a, b, c)

        # 输出结果
        return c.asnumpy()
    # elif operator_name == "prod":
    #     B = topi.prod(A)
    # elif operator_name == "isnan":
    #     B = topi.isnan(A)
    
    else:
        raise ValueError(f"Unsupported operator: {operator_name}")
    
    # 创建schedule
    
    s = te.create_schedule(B.op)

    

    # 编译生成可执行的模块
    f = tvm.build(s, [A, B], "llvm")

    # 创建TVM的context，这里使用CPU
    ctx = tvm.cpu(0)
    
    # 创建TVM NDArray
    a = tvm.nd.array(input_data, ctx)
    

    if operator_name == "max":
        b = tvm.nd.array(np.zeros(shape[0], dtype=input_data.dtype), ctx)
        f(a, b)
        print("----------------")
        print("a-asnumpyt")
        print(a.asnumpy())
        print("b-asnumpyt")
        print(b.asnumpy())
        print("----------------")
        return b.asnumpy()
    else:
    # 创建输出存储空间
        b = tvm.nd.array(np.zeros(shape, dtype=input_data.dtype), ctx)
    
    # 执行计算
    f(a, b)

    # 返回计算结果
    return b.asnumpy()

# 示例使用

def loaded_module_run(loaded_module, input_data,name=None):
    shape = input_data.shape
    # 创建TVM的context，这里使用CPU
    ctx = tvm.cpu(0)
    
    # 创建TVM NDArray
    a = tvm.nd.array(input_data, ctx)
    # 创建输出存储空间
    b = tvm.nd.array(np.zeros(shape, dtype=input_data.dtype), ctx)

    if(name == "max"):
        print("max operator Troubleshooting the error: ",name)
        print(shape)
        
        b = tvm.nd.array(np.zeros(shape[0], dtype=input_data.dtype), ctx)
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        print("init data") 
        print(a.asnumpy())
        # print(name)
        loaded_module(a,b)
        print("after load")
        print(b.asnumpy())
        print(a.asnumpy())
        # print(a.asnumpy())
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        return b.asnumpy()
    elif (name == "adaptive_pool_avg"):
        print("adaptive_pool_avg runtime operator Troubleshooting the error: ",name)
        print(shape)
        b = tvm.nd.array(np.zeros((8,8), dtype=input_data.dtype), ctx)
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        print("init data") 
        print(a.asnumpy())
        # print(name)
        loaded_module(a,b)
        print("after load")
        print(b.asnumpy())
        print(a.asnumpy())
        # print(a.asnumpy())
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        return b.asnumpy()
    elif (name == "add"):
        b = tvm.nd.array(input_data, ctx)
        c = tvm.nd.array(np.zeros(shape, dtype=input_data.dtype), ctx)
        loaded_module(a,b,c)
        return c.asnumpy()
    
    loaded_module(a, b)
    
    # 返回计算结果
    return b.asnumpy()

# 初始化一个空字典来记录算子通过测试的次数
operators_count = {}

def load_and_execute_from_ll(directory):
    """
    遍历指定目录下的所有.ll文件，并加载执行。
    """
    passed_count = 0
    failed_count = 0
    failed_files = []
    
    # 确保目录存在
    if not os.path.exists(directory):
        print(f"目录 {directory} 不存在。")
        return
    
    # 遍历目录下的所有.ll文件
    for filename in os.listdir(directory):
        if filename.endswith('.ll'):
            file_path = os.path.join(directory, filename)
            # 假设 tvm.runtime.load_from_ll 是一个有效的函数
            # 注意: 实际中可能需要替换为适合你的环境和版本的函数
            try:
                rt_m = tvm.runtime.load_from_ll(file_path, "ll")
                
                operator, shape = parse_filename(filename)

                input_data = np.random.uniform(-1, 1, size=shape).astype("float32")
                if(operator == "max"):
                    result_by_runtime_module = loaded_module_run(rt_m, input_data, operator)
                else:
                    result_by_runtime_module = loaded_module_run(rt_m, input_data,operator)
                golden_data = compute_with_operator(operator, input_data)
                tvm.testing.assert_allclose(golden_data, result_by_runtime_module, rtol=1e-5)
                print(f"算子名称: {operator}, Shape信息: {shape} pass test")

                if operator in operators_count:
                    operators_count[operator] += 1
                else:
                    operators_count[operator] = 1
                passed_count += 1
            except Exception as e:
                print(f"算子 {filename} 未通过测试: {e}")
                failed_count += 1
                failed_files.append(filename)

    print(f"总共通过了 {passed_count} 个文件，未通过了 {failed_count} 个文件。")
    print("未通过测试的文件名:")
    for failed_file in failed_files:
        print(failed_file)

# 指定你的.ll文件所在目录
ll_directory = 'nn_ll'
load_and_execute_from_ll(ll_directory)

# 打印每个算子出现的次数
for operator, count in operators_count.items():
    print(f"{operator}: {count} 次")