
import torch
import time

# Modified bger function (the original function from util.py)
def bger(x, y):
    return x.unsqueeze(2).bmm(y.unsqueeze(1))

# Optimized bger function
def optimized_bger(u, v):
    return torch.einsum('bi,bj->bij', u, v)

# Benchmarking setup and results are provided as functions so that they can be executed if needed


if __name__ == "__main__":
    num_iterations = 1000
    test_cases = [
        (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[2.0, 3.0], [4.0, 5.0]])),
        (torch.tensor([[1.0], [3.0]]), torch.tensor([[2.0], [4.0]])),
        (torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([[2.0, 3.0, 4.0]]))
    ]

    for u, v in test_cases:
        # Benchmarking for each method
        for method in [bger, optimized_bger]:
            if method == bger:
                start_time = time.time()
                for _ in range(num_iterations):
                    result_original = method(u, v)
                end_time = time.time()
                execution_time_original = (end_time - start_time) / num_iterations
            else:
                start_time = time.time()
                for _ in range(num_iterations):
                    result_optimized = method(u, v)
                end_time = time.time()
                execution_time_optimized = (end_time - start_time) / num_iterations
                print(f"Method: bdiag, Time: {execution_time_original:.6f} [sec] \n"
                      f"Method: {method.__name__}, Time: {execution_time_optimized:.6f} [sec] \n"
                      f"Equal: {torch.allclose(result_optimized, result_original)}")
