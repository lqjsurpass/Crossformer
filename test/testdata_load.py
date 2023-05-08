import numpy as np
import torch
import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence,pad_sequence

#c4 232581  c1 254943  c6 225119
dataset = np.load("datasetc6.npy")
datasetmask = np.load("datasetmaskc6.npy")
print(dataset)
print(datasetmask)