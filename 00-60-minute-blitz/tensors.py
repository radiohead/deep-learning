from __future__ import print_function
import torch
import numpy as np

# Basics
z = torch.zeros(5, 3, dtype=torch.long)
print(z)

m = torch.tensor([5.5, 3], dtype=torch.float16)
print(m)

n = m.new_ones(5, 3)
print(n)

# Addition
x = torch.empty(5, 3, dtype=torch.float64)
print(x)

y = torch.rand(5, 3, dtype=torch.float64)
print(y)

v = torch.randn_like(y, dtype=torch.float32)
print(v)
print(v.size())

t = y + v
print(y + v)
print(torch.add(y, v))

torch.add(y, v, out=x)
print(x)
print(y.add_(x))

# Slicing
r = torch.randn(4, 3)
print(r)
print(r[:, 1:])

g = torch.randn(4, 4)
h = g.view(16)
i = g.view(-1, 8)
print(g.size(), h.size(), i.size())

w = torch.rand(1)
print(w, w.item())

a = torch.ones(5)
print(a, a.numpy())

ones = np.ones(5)
tensor = torch.from_numpy(ones)
print(ones, tensor)

np.add(ones, 5, out=ones)
print(ones, tensor)

print(torch.cuda.is_available())
print(torch.ones(3, 3, device='cpu'))
