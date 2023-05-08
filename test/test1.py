import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np

tensor1 = torch.arange(0,24).view(2,3,4).float()
tensor2 = tensor1.mean(dim = 1, keepdim = True)
print(tensor1)
print(tensor2)

tensor1 = torch.arange(0,24).view(2,3,4)
tensor2 = torch.arange(0,24).view(2,3,4)
tensor3 = torch.cat((tensor1,tensor2),dim=1)
print(tensor3)

tensor5 = tensor1[:, :1, :]
print("tensor5:",tensor5)

tensor6 = tensor5.expand(-1, 4, -1)
print("tensor6:",tensor6)

tensor4 = tensor1[:, :1, :].expand(-1, 4, -1)
print(tensor4)
