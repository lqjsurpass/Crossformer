import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
a = np.arange(0,12)
print(a)
a = a.reshape(3,4)
print(a)
print("a[:]:",a[:])
print("a[:]:",a[1:2])
print("a[:]2:",a[1:2:])

print("a[::]:",a[::])
print("a[2::]:",a[2::])

b = np.arange(0,48).reshape(6,8)
print(b)

print("b[::3],",b[::2])

print("b[::3],",b[1::2])

print("b[::3]1,",b[:,1:4:2])

tensor1 = torch.tensor(b).float()
print(tensor1)
tensor2 = torch.max_pool1d(tensor1,3,3)
print(tensor2)