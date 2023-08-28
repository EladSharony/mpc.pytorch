import torch
from torch.autograd import Variable


def get_data_maybe(x):
    return x if not isinstance(x, Variable) else x.data

# Jacobian calculation methods
def jacobian(f, x, eps=1e-5):
    if x.ndimension() == 2:
        assert x.size(0) == 1
        x = x.squeeze()

    e = Variable(torch.eye(len(x)).type_as(get_data_maybe(x)))
    J = []
    for i in range(len(x)):
        J.append((f(x + eps*e[i]) - f(x - eps*e[i]))/(2.*eps))
    J = torch.stack(J).transpose(0,1)
    return J

def jacobian_optimized_high_precision(f, x, eps=1e-5):
    x_high_precision = x.double()
    n = x_high_precision.numel()
    e = torch.eye(n, dtype=torch.float64, device=x_high_precision.device)
    f_original = f(x_high_precision)
    J_separate = []
    for i in range(n):
        perturbed_value = f(x_high_precision + eps * e[i])
        for j, f_val in enumerate(perturbed_value.unbind()):
            J_separate.append((f_val - f_original[j]) / eps)
    J_high_precision = torch.stack(J_separate).reshape(n, -1).T
    return J_high_precision.to(dtype=x.dtype)

def jacobian_optimized(f, x, eps=1e-5):
    n = x.numel()
    e = torch.eye(n, device=x.device)
    f_original = f(x)
    J_separate = []
    for i in range(n):
        perturbed_value = f(x + eps * e[i])
        for j, f_val in enumerate(perturbed_value.unbind()):
            J_separate.append((f_val - f_original[j]) / eps)
    J = torch.stack(J_separate).reshape(n, -1).T
    return J

# Analytical Jacobian methods
def linear_function_analytical_jacobian(x):
    return torch.tensor([[2, 3], [4, -1]], dtype=x.dtype, device=x.device)

def exponential_function_analytical_jacobian(x):
    return torch.diag(torch.exp(x))

def trigonometric_function_analytical_jacobian(x):
    return torch.tensor([[torch.cos(x[0]), 0], [0, -torch.sin(x[1])]], dtype=x.dtype, device=x.device)

def polynomial_function_analytical_jacobian(x):
    return torch.tensor([
        [3*x[0]**2 + 2*x[0]*x[1], x[0]**2],
        [x[1]**2, 2*x[0]*x[1] + 3*x[1]**2]
    ], dtype=x.dtype, device=x.device)

# Benchmarking
if __name__ == "__main__":
    x_point = torch.tensor([1.0, 2.0])
    num_iterations = 1000
    import time

    def linear_function(x):
        return torch.stack([2*x[0] + 3*x[1], 4*x[0] - x[1]])

    def exponential_function(x):
        return torch.exp(x)

    def trigonometric_function(x):
        return torch.stack([torch.sin(x[0]), torch.cos(x[1])])

    def polynomial_function(x):
        return torch.stack([x[0]**3 + x[0]**2*x[1], x[0]*x[1]**2 + x[1]**3])


    sample_functions = [linear_function, exponential_function, trigonometric_function, polynomial_function]
    methods = [jacobian, jacobian_optimized_high_precision, jacobian_optimized]

    # Print the table header
    header = "Function".ljust(25) + "Method".ljust(35) + "Time [sec]".ljust(15) + "Result"
    print(header)
    print('-' * 100)

    for func in sample_functions:
        for method in methods:
            # Benchmarking for each method
            start_time = time.time()
            for _ in range(num_iterations):
                result = method(func, x_point)
            end_time = time.time()
            execution_time = (end_time - start_time) / num_iterations

            # Print results in table format
            row = func.__name__.ljust(25) + method.__name__.ljust(35) + f"{execution_time:.6f}".ljust(15) + f"{result.flatten()}"
            print(row)

        # Separate each function's results for better readability
        print('-' * 100)
