import tvm
import tvm.testing as testing
from tvm import te
from tvm import topi
import numpy as np

def test_llvm_c():
    target = tvm.target.Target(target="c", host="llvm")
    dev = tvm.device(target.kind.name, 0)

    n = 64

    A = te.placeholder((n,))
    B = topi.abs(A)
    s = te.create_schedule(B.op)

    A.set_name("A")
    B.set_name("B")

    func = tvm.build(s, [A, B], target=target)

    print(tvm.lower(s, [A, B], simple_mode=True))
    print("\n", func.imported_modules[0].get_pure_source())

    # a = tvm.nd.array(np.ones((n,)))
    # b = tvm.nd.array(np.zeros((n,)))
    # func(a, b)
    # print(a, b)

if __name__ == "__main__":
    test_llvm_c()
