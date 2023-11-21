import tvm
from tvm import te
import numpy as np

def test_cuda_kernel():
    dtype = "float32"
    target = "cuda"
    tgt = tvm.target.Target(target=target, host="c")

    # Compute declaration
    N = 128
    A = te.placeholder((N, N), name="A")
    B = te.placeholder((N, N), name="B")
    k = te.reduce_axis((0, N), name="k")
    C = te.compute((N, N), lambda i, j: te.sum(A[i][k] * B[k][j], axis=[k]), name="C")

    # Schedule
    s = te.create_schedule([C.op])
    CC = s.cache_write(C, "local")
    i, j = s[C].op.axis
    bx, tx, ii, ji = s[C].tile(i, j, 1, 2)
    s[C].bind(bx, te.thread_axis("blockIdx.x"))
    s[C].bind(tx, te.thread_axis("threadIdx.x"))
    s[C].vectorize(ji)
    s[CC].compute_at(s[C], tx)
    i, j = s[CC].op.axis
    k = s[CC].op.reduce_axis[0]
    ko, ki = s[CC].split(k, 2)
    s[CC].unroll(ki)
    s[CC].vectorize(j)

    func_tvm = tvm.build(s, [A, B, C], target=tgt)
    print(func_tvm.imported_modules[0].get_source())
    # print(func_tvm.get_source())

def test_c_kernel():
    dtype = "float32"
    target = "c"
    tgt = tvm.target.Target(target=target, host="c")

    # Compute declaration
    N = 128
    A = te.placeholder((N, N), name="A")
    B = te.placeholder((N, N), name="B")
    k = te.reduce_axis((0, N), name="k")
    C = te.compute((N, N), lambda i, j: te.sum(A[i][k] * B[k][j], axis=[k]), name="C")

    # Schedule
    s = te.create_schedule([C.op])
    i, j = s[C].op.axis
    bx, tx, ii, ji = s[C].tile(i, j, 1, 2)
    s[C].vectorize(ji)

    func_tvm = tvm.build(s, [A, B, C], target=tgt)
    print(func_tvm.imported_modules[0].get_source())
    # print(func_tvm.get_source())


if __name__ == "__main__":
    test_cuda_kernel()
    test_c_kernel()