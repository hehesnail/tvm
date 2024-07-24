import os
import torch
import tvm
import numpy as np
import tvm.testing as testing
from tvm import te
from tvm import topi
from tvm.runtime import load_module

def test_omp():
    """
    test case for omp pragama, cext module functionality
    """
    target = tvm.target.Target(target="c", host="llvm")
    
    M = 256
    K = 256
    N = 256
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")
    s = te.create_schedule(C.op)

    # Blocking by loop tiling
    bn = 32
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    (k,) = s[C].op.reduce_axis
    ko, ki = s[C].split(k, factor=4)
    # re-ordering
    s[C].reorder(xo, yo, ko, xi, ki, yi)
    s[C].parallel(xo)

    func = tvm.build(s, [A, B, C], target=target)

    # print(func.get_source())
    # print(func.imported_modules[0].get_source())
    print(tvm.lower(s, [A, B, C], simple_mode=True))
    print("\n", func.imported_modules[0].get_pure_source())

    dev = tvm.device(target.kind.name, 0)
    np_a = np.ones((M, K)).astype(A.dtype) * 3
    np_b = np.ones((K, N)).astype(B.dtype) * 2
    np_c = np.zeros((M, N), dtype=C.dtype)
    a = tvm.nd.array(np_a, dev)
    b = tvm.nd.array(np_b, dev)
    c = tvm.nd.array(np_c, dev)

    func(a, b, c)
    print(c, c.numpy().shape)

    torch_a = torch.from_numpy(np_a)
    torch_b = torch.from_numpy(np_b)
    torch_c = torch.matmul(torch_a, torch_b)
    print(torch_c, torch_c.numpy().shape)

    testing.assert_allclose(c.numpy(), torch_c.numpy())


def test_load_c_source():
    """
    test case for load c code to cext module & run
    """
    target = tvm.target.Target(target="c", host="llvm")
    M = 4
    K = 4
    N = 4
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = topi.nn.matmul(A, B)
    s = te.create_schedule(C.op)
    func = tvm.build(s, [A, B, C], target=target)

    # note: function name matters, if not given, func name needs to be default_function_kernel
    c_code = """
    void default_function_kernel(float* A, float* B, float* C) {
    for (int32_t i0 = 0; i0 < 4; ++i0) {
        for (int32_t i1 = 0; i1 < 4; ++i1) {
        C[((i0 * 4) + i1)] = 0.000000e+00f;
        for (int32_t k = 0; k < 4; ++k) {
            C[((i0 * 4) + i1)] = (C[((i0 * 4) + i1)] + (A[((i0 * 4) + k)] * B[((k * 4) + i1)]));
        }
        }
    }
    }
    """
    print(func.imported_modules[0].get_source())
    # note: use set_source to set device kernel code for cext module.
    # note: this only change code in cext module, clean_code still not changed
    func.imported_modules[0].set_source(c_code)
    print(func.imported_modules[0].get_source())

    dev = tvm.device(target.kind.name, 0)
    np_a = np.ones((M, K)).astype(A.dtype) * 3
    np_b = np.ones((K, N)).astype(B.dtype) * 2
    np_c = np.zeros((M, N), dtype=C.dtype)
    a = tvm.nd.array(np_a, dev)
    b = tvm.nd.array(np_b, dev)
    c = tvm.nd.array(np_c, dev)

    func(a, b, c)
    print(c, c.numpy().shape)

    torch_a = torch.from_numpy(np_a)
    torch_b = torch.from_numpy(np_b)
    torch_c = torch.matmul(torch_a, torch_b)
    print(torch_c, torch_c.numpy().shape)

    testing.assert_allclose(c.numpy(), torch_c.numpy())


def test_cuda():
    target = tvm.target.Target(target="cuda", host="llvm")
    dev = tvm.device(target.kind.name, 0)

    n = 64
    num_thread = 8
    tx = te.thread_axis("threadIdx.x")
    bx = te.thread_axis("blockIdx.x")

    A = te.placeholder((n,), name="A")
    B = te.compute((n,), lambda i: A[i] + tvm.tir.const(1, A.dtype), name="B")
    s = te.create_schedule(B.op)
    xo, xi = s[B].split(B.op.axis[0], factor=num_thread)
    s[B].bind(xo, bx)
    s[B].bind(xi, tx)

    cuda_func = tvm.build(s, [A, B], target=target)
    # note: imported device module impls get_pure_source() api to get clean code 
    print(cuda_func.imported_modules[0].get_pure_source())
    
    a = tvm.nd.array(np.ones(n).astype(A.dtype), dev)
    b = tvm.nd.array(np.zeros((n), dtype=A.dtype), dev)
    print(a, b)
    cuda_func(a, b)
    print(b)


def test_cext_module_save():
    """
    test case for cext module save & load & test
    """
    target = tvm.target.Target(target="c", host="llvm")
    M, K, N = 4, 4, 4
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = topi.nn.matmul(A, B)
    s = te.create_schedule(C.op)
    func = tvm.build(s, [A, B, C], target=target)

    host_module = func
    device_module = func.imported_modules[0]

    save_path = "temp_saved_modules"
    host_path = "host_module"
    device_path = "device_module"

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path, host_path))
        os.mkdir(os.path.join(save_path, device_path))

    host_module.save(os.path.join(save_path, host_path, "topology_expansion_[['add', 'cos', 'asin', 'ceil']]_[[12, 1, 18], [16, 3, 1], [12, 1, 18]]"), "ll")
    device_module.save(os.path.join(save_path, device_path, "topology_expansion_[['add', 'cos', 'asin', 'ceil']]_[[12, 1, 18], [16, 3, 1], [12, 1, 18]]"), "c")

def test_cext_module_load():
    target = tvm.target.Target(target="c", host="llvm")
    save_path = "temp_saved_modules"
    if not os.path.exists(save_path):
        raise Exception(f"{save_path} not found")

    host_module = load_module(os.path.join(save_path, "host_module"), "ll")
    device_module = load_module(os.path.join(save_path, "device_module"), "c")
    host_module.import_module(device_module)

    print(host_module.imported_modules[0].get_source())
    # note: function name matters, if not given, func name needs to be default_function_kernel
    c_code = """
    void default_function_kernel(float* A, float* B, float* C) {
    for (int32_t i0 = 0; i0 < 4; ++i0) {
        for (int32_t i1 = 0; i1 < 4; ++i1) {
        C[((i0 * 4) + i1)] = 0.000000e+00f;
        for (int32_t k = 0; k < 4; ++k) {
            C[((i0 * 4) + i1)] = (C[((i0 * 4) + i1)] + (A[((i0 * 4) + k)] * B[((k * 4) + i1)]));
        }
        }
    }
    }
    """
    host_module.imported_modules[0].set_source(c_code)
    print(host_module.imported_modules[0].get_source())

    dev = tvm.device(target.kind.name, 0)
    M, K, N = 4, 4, 4
    # dtype非常重要，需要和生成时的数据类型匹配
    np_a = np.ones((M, K)).astype("float32") * 3  
    np_b = np.ones((K, N)).astype("float32") * 2
    np_c = np.zeros((M, N)).astype("float32")
    a = tvm.nd.array(np_a, dev)
    b = tvm.nd.array(np_b, dev)
    c = tvm.nd.array(np_c, dev)

    host_module(a, b, c)
    print(c, c.numpy().shape)

    torch_a = torch.from_numpy(np_a)
    torch_b = torch.from_numpy(np_b)
    torch_c = torch.matmul(torch_a, torch_b)
    print(torch_c, torch_c.numpy().shape)

    testing.assert_allclose(c.numpy(), torch_c.numpy())

if __name__ == "__main__":
    # test_omp()
    # test_load_c_source()
    # test_cuda()
    test_cext_module_save()
    # test_cext_module_load()