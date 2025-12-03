import torch
import time

t1 = torch.rand(1000, 1000)
t2 = torch.rand(1000, 1000)

print("Starting")
start = time.time()
t1.matmul(t2)
print("Matmul time (CPU): ", time.time() - start)

t1 = t1.to("cuda")
t2 = t2.to("cuda")

start = time.time()
t1.matmul(t2)
print("Matmul time (GPU): ", time.time() - start)
