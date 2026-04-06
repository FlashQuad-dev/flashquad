import sys
import os
import cupy as cp
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))# to parent dir  
from flashquad import trapz_integrate, simpson_integrate, booles_integrate, gauss_integrate, mc_integrate, adpmc_integrate
from flashquad import set_backend, get_data_type

data_type=cp.float32
set_backend(data_type)

def function(x, y, z, params):
    a = params[0]
    b = params[1]
    c = params[2]
    return a * cp.exp(-b * (x**2 + y**2 + z**2)) + c * cp.sin(x) * cp.cos(y) * cp.exp(z)

def boundary(x, y, z):
    return x**2 + y**2 + z**2 < 10

a_values = cp.linspace(1.0, 10.0, 10000, dtype=data_type)
b_values = cp.linspace(2.0, 20.0, 10000, dtype=data_type)
c_values = cp.linspace(0.5, 5, 10000, dtype=data_type)
param_values = cp.stack((a_values, b_values, c_values), axis=1) 

bound = [[0, 1], [0, 1], [0, 1]]
num_point = [33, 33, 33]

start_time = time.time()
integral_values = trapz_integrate(function, param_values, bound, num_point, boundary)
end_time = time.time()
elapsed_time = end_time - start_time

print(integral_values)
print("time used: " + str(elapsed_time) + "s")
print(integral_values.dtype)
print(integral_values.device)
print(a_values.dtype)
