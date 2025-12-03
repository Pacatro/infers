import torch

m = 2
k = 3
n = 4

t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
t2 = torch.tensor([[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

print(t1.shape)
print(t2.shape)
t3 = t1 @ t2
print(t3.shape)
print(t3)
