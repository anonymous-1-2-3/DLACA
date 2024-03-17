import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import maxabs_scale
import os
from torch.utils.data import DataLoader
import platform
from scipy.fft import fft


class PU_DataSet(Dataset):
    def __init__(self, path, snr=-10, FFT=False, seed=None):
        self.path = path
        self.snr = snr
        self.FFT = FFT
        self.file_list = self.get_all_sample()
        self.split_sys = self.get_split_sys()
        self.seed = seed

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        data = self.data_preprocess(data, self.snr, self.FFT)
        fault, work_cond = self.label_fault_work(self.file_list[idx])
        fault = torch.tensor(fault).type(torch.LongTensor)
        work_cond = torch.tensor(work_cond).type(torch.LongTensor)
        return data, fault, work_cond

    def data_preprocess(self, x, snr, FFT):
        x = x.astype(np.float32)
        np.random.seed(self.seed)
        if snr < 0:
            snr = 10 ** (snr / 10.0)
            xpower = np.sum(x ** 2) / len(x)
            npower = xpower / snr
            noise = np.random.randn(len(x)) * np.sqrt(npower)
            noise = noise.astype(np.float32)
            x = x + noise
        if FFT:
            fft_x = fft(x)
            amp_x = abs(fft_x) / len(x) * 2
            x = amp_x[0:int(len(x) / 2)]
        x = maxabs_scale(x)
        x = torch.from_numpy(x)
        return x

    def get_all_sample(self):
        file_path_list = []
        for one_file_path in self.path:
            for root, dirs, files in os.walk(one_file_path):
                if dirs == []:
                    for file in files:
                        file_path_list.append(os.path.join(root, file))
        return file_path_list

    def label_fault_work(self, path):
        global fault
        global work_cond
        file_flag = path.split(self.split_sys)
        file_name = file_flag[-1]
        if 'NC' in file_name:
            fault = 0
        elif 'MR' in file_name:
            fault = 7
        elif 'IR' in file_name:
            fault_flag = file_flag[-2]
            if fault_flag == 'I0':
                fault = 1
            if fault_flag == 'I1':
                fault = 2
            if fault_flag == 'I2':
                fault = 3
        elif 'OR' in file_name:
            fault_flag = file_flag[-2]
            if fault_flag == 'O0':
                fault = 4
            if fault_flag == 'O1':
                fault = 5
            if fault_flag == 'O2':
                fault = 6

        if 'N15_M07_F10' in file_name:
            work_cond = int(0)
        elif 'N09_M07_F10' in file_name:
            work_cond = int(1)
        elif 'N15_M01_F10' in file_name:
            work_cond = int(2)
        elif 'N15_M07_F04' in file_name:
            work_cond = int(3)
        return fault, work_cond

    def get_split_sys(self):
        if platform.system().lower() == 'windows':
            split_sym = '\\'
        else:
            split_sym = '/'
        return split_sym


class SQ_DataSet(Dataset):
    def __init__(self, path, snr=-10, FFT=False, seed=None):
        self.path = path
        self.snr = snr
        self.FFT = FFT
        self.file_list = self.get_all_sample()
        self.split_sys = self.get_split_sys()
        self.seed = seed

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        data = self.data_preprocess(data, self.snr, self.FFT)
        fault, work_cond = self.label_fault_work(self.file_list[idx])
        fault = torch.tensor(fault).type(torch.LongTensor)
        work_cond = torch.tensor(work_cond).type(torch.LongTensor)
        return data, fault, work_cond

    def data_preprocess(self, x, snr, FFT):
        x = x.astype(np.float32)
        np.random.seed(self.seed)
        if snr < 0:
            snr = 10 ** (snr / 10.0)
            xpower = np.sum(x ** 2) / len(x)
            npower = xpower / snr
            noise = np.random.randn(len(x)) * np.sqrt(npower)
            noise = noise.astype(np.float32)
            x = x + noise
        if FFT:
            fft_x = fft(x)
            amp_x = abs(fft_x) / len(x) * 2
            x = amp_x[0:int(len(x) / 2)]
        x = maxabs_scale(x)
        x = torch.from_numpy(x)
        return x

    def get_all_sample(self):
        file_path_list = []
        for one_file_path in self.path:
            for root, dirs, files in os.walk(one_file_path):
                if dirs == []:
                    for file in files:
                        file_path_list.append(os.path.join(root, file))
        return file_path_list

    def label_fault_work(self, path):
        global fault
        file_flag = path.split(self.split_sys)
        file_name = file_flag[-1]
        if 'NC' in file_name:
            fault = 0
        elif 'I0' in file_name:
            fault = 1
        elif 'I1' in file_name:
            fault = 2
        elif 'I2' in file_name:
            fault = 3
        elif 'O0' in file_name:
            fault = 4
        elif 'O1' in file_name:
            fault = 5
        elif 'O2' in file_name:
            fault = 6

        work_flag = file_name.split('_')[2]
        work_cond = int(work_flag)
        return fault, work_cond

    def get_split_sys(self):
        if platform.system().lower() == 'windows':
            split_sym = '\\'
        else:
            split_sym = '/'
        return split_sym


def Task_DataSet(dataname):
    Fault_DataSet = None
    if dataname == 'PU':
        Fault_DataSet = PU_DataSet
    if dataname == 'SQ':
        Fault_DataSet = SQ_DataSet
    return Fault_DataSet