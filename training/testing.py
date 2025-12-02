import torch

t = torch.rand(2, 2)
t_flatten = t.flatten()
print(t.shape)
print(t_flatten.shape)
