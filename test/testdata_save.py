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


def getData1(root_path):
    list1 = []
    for i in range(1,316):
        df_raw = pd.read_csv(os.path.join(root_path, 'c_6_'+format(i,"03")+'.csv'))
        list1.append(torch.tensor(df_raw.to_numpy()))
    return list1

def getData(root_path):
    result = getData1(root_path)

    padx = pad_sequence(result, batch_first=True, padding_value=0)
    mask = (padx != 0)
    return padx.to(torch.float32),mask
#数据预处理 1.读取 c1 c4 c6中的文件 分别存储为npy
dataset1,mask1 = getData("../data/PHM/c6/")

# np.save("dataset1.npy",dataset1)
# np.save("datasetmask1.npy",mask1)
# datasetmask = np.load("datasetmask1.npy")
datasetmask = mask1[:,:,0]
np.save("datasetc6.npy",dataset1)
np.save("datasetmaskc6.npy",datasetmask)



# np.save("datasetmask.npy",datasetmask)
