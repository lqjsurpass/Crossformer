import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from utils.tools import StandardScaler

import warnings
warnings.filterwarnings('ignore')

class Dataset_MTS(Dataset):
    def __init__(self, root_path, data_path='ETTh1.csv', flag='train', size=None, 
                  data_split = [0.7, 0.1, 0.2], scale=True, scale_statistic=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.in_len = size[0]
        self.out_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.scale = scale
        #self.inverse = inverse
        
        self.root_path = root_path
        self.data_path = data_path
        self.data_split = data_split
        self.scale_statistic = scale_statistic
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        if (self.data_split[0] > 1):
            train_num = self.data_split[0]; val_num = self.data_split[1]; test_num = self.data_split[2];
        else:
            train_num = int(len(df_raw)*self.data_split[0]); 
            test_num = int(len(df_raw)*self.data_split[2])
            val_num = len(df_raw) - train_num - test_num; 
        border1s = [0, train_num - self.in_len, train_num + val_num - self.in_len]
        border2s = [train_num, train_num+val_num, train_num + val_num + test_num]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            if self.scale_statistic is None:
                self.scaler = StandardScaler()
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
            else:
                self.scaler = StandardScaler(mean = self.scale_statistic['mean'], std = self.scale_statistic['std'])
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.in_len
        r_begin = s_end
        r_end = r_begin + self.out_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y
    
    def __len__(self):
        return len(self.data_x) - self.in_len- self.out_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_PHM2(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=True,):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.in_len = size[0]
            self.out_len = size[1]
            # self.seq_len = size[0]
            # self.label_len = size[1]
            # self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    # self.set_type  0 1 2
    # type_map = {'train': 0, 'val': 1, 'test': 2}


    def __read_data__(self):
        self.scaler = StandardScaler()

        # data_x = np.load((os.path.join(self.root_path,
        #                                   'trandaset_x.npy')))
        # data_y = np.load((os.path.join(self.root_path,
        #                                   'trandaset_y.npy')))
        train_data_x = np.load((os.path.join(self.root_path,'trandaset1_x.npy')))
        train_data_y = np.load((os.path.join(self.root_path, 'trandaset1_y.npy')))
        test_data_x = np.load((os.path.join(self.root_path,'testdaset1_x.npy')))
        test_data_y = np.load((os.path.join(self.root_path,'testdaset1_y.npy')))

        data_x = np.concatenate([train_data_x,test_data_x],axis=0)
        data_y = np.concatenate([train_data_y,test_data_y],axis=0)
        if self.scale:
            self.scaler.fit(data_x)
            data_x = self.scaler.transform(data_x)
            train_data_x = self.scaler.transform(train_data_x)
            test_data_x = self.scaler.transform(test_data_x)

            self.scaler.fit(data_y)
            data_y = self.scaler.transform(data_y)
            train_data_y = self.scaler.transform(train_data_y)
            test_data_y = self.scaler.transform(test_data_y)
        else:
            #不标准化 什么都不做
            data_x = data_x
            data_y = data_y

        num_batch = train_data_x.shape[0]
        num_train = int(num_batch * 0.9)
        num_vali = num_batch - num_train
        train_x = train_data_x[0:num_train]
        train_y = train_data_y[0:num_train]
        vali_x = train_data_x[num_train:]
        vali_y = train_data_y[num_train:]
        test_x = test_data_x
        test_y = test_data_y
        # num_batch = data_x.shape[0]
        # num_train = int(num_batch * 0.7)
        # num_test = int(num_batch * 0.2)
        # num_vali = num_batch - num_train - num_test
        #
        # train_x = data_x[0:num_train]
        # train_y = data_y[0:num_train]
        # test_x = data_x[num_train:num_train+num_test]
        # test_y = data_y[num_train:num_train+num_test]
        # vali_x = data_x[num_train+num_test:]
        # vali_y = data_y[num_train+num_test:]

        self.data_all_x = [train_x,vali_x,test_x]
        self.data_all_y = [train_y,vali_y,test_y]
        self.data_x = self.data_all_x[self.set_type]
        self.data_y = self.data_all_x[self.set_type]

        self.len = self.data_x.shape[0]



    def __getitem__(self, index):

        seq_x = self.data_x[index]
        seq_y = self.data_y[index]
        return seq_x, seq_y

    def __len__(self):
        return self.len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
