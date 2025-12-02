import torch
from time import time

start = time()
t1 = torch.rand(700, 700, 700)
t2 = torch.rand(700, 700, 700)
print(f"Duration: {time() - start:.6f}s")
t1_gpu = t1.to("cuda")
t2_gpu = t2.to("cuda")

print("Running on CPU")
start = time()
t3 = t1 + t2
print(f"Duration: {time() - start:.6f}s")

print("Running on CUDA")
start = time()
t3 = t1_gpu + t2_gpu
print(f"Duration: {time() - start:.6f}s")
