import torch
import time


def test_matmul_time():
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
    t1 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    t2 = torch.tensor([[[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    t4 = torch.rand(3, 2, 2)
    print(t1.shape)
    print(t2.shape)
    print(t4.shape)
    t3 = t1.matmul(t4)
    print(t3.shape)
    print(t3)


t = torch.rand(3, 2, 2)
t_t = t.T
print(t.shape)
print(t_t.shape)
