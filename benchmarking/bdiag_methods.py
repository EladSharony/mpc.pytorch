import torch
import time
from torch.autograd import Variable

def bdiag(d):
    assert d.ndimension() == 2
    nBatch, sz = d.size()
    dtype = d.type() if not isinstance(d, Variable) else d.data.type()
    D = torch.zeros(nBatch, sz, sz).type(dtype)
    I = torch.eye(sz).repeat(nBatch, 1, 1).type(dtype).byte()
    D[I] = d.view(-1)
    return D


# Optimized bdiag function
def optimized_bdiag(d):
    n_batch, n_dim = d.size()
    D = torch.zeros(n_batch, n_dim, n_dim, dtype=d.dtype, device=d.device)
    for i in range(n_dim):
        D[:, i, i] = d[:, i]
    return D


# Benchmarking setup and results are provided as functions so that they can be executed if needed

if __name__ == "__main__":
    num_iterations = 1000
    test_cases = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        torch.tensor([[1.0]])
    ]

    for test in test_cases:
        # Benchmarking for each method
        for method in [bdiag, optimized_bdiag]:
            if method == bdiag:
                start_time = time.time()
                for _ in range(num_iterations):
                    result_original = method(test)
                end_time = time.time()
                execution_time_original = (end_time - start_time) / num_iterations
            else:
                start_time = time.time()
                for _ in range(num_iterations):
                    result_optimized = method(test)
                end_time = time.time()
                execution_time_optimized = (end_time - start_time) / num_iterations
                print(f"Method: bdiag, Time: {execution_time_original:.6f} [sec] \n"
                      f"Method: {method.__name__}, Time: {execution_time_optimized:.6f} [sec] \n"
                      f"Equal: {torch.allclose(result_optimized, result_original)}")
