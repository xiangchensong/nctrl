import os
import glob
import torch
import random
import json
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pytorch_lightning as pl

class ARHMNLICADataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.path = Path(data_path)
        self.raw_data = pickle.load(open(self.path/"data.pkl", "rb"))
        self.meta = json.load(open(self.path/"meta.json", "r"))
        self.data = {}
        self.z = torch.Tensor(self.raw_data["Z"]) # (n_samples, n_time_steps, n_latent)
        self.x = torch.Tensor(self.raw_data["X"])  # (n_samples, n_time_steps, n_features)
        self.c = torch.LongTensor(self.raw_data["C"]) # (n_samples, n_time_steps)
        self.A = torch.Tensor(self.raw_data["A"])
        # self.meta = self.raw_data["meta"]
    def __len__(self):
        return len(self.x) 
    
    def __getitem__(self, idx):
        x_t = self.x[idx]
        z_t = self.z[idx]
        c_t = self.c[idx]
        return x_t, z_t, c_t

class SimulationDatasetTSTwoSampleNS(Dataset):
    
    def __init__(self, data_path, c=False):
        super().__init__()
        self.path = Path(data_path)
        self.raw_data = pickle.load(open(self.path/"data.pkl", "rb"))
        self.meta = json.load(open(self.path/"meta.json", "r"))
        self.data = {}
        self.z = torch.Tensor(self.raw_data["Z"]) # (n_samples, n_time_steps, n_latent)
        self.x = torch.Tensor(self.raw_data["X"]) # (n_samples, n_time_steps, n_features)
        if c:
            self.c = torch.Tensor(self.raw_data["C"]) # (n_samples, n_time_steps)
        else:
            self.c = torch.zeros_like(torch.Tensor(self.raw_data["C"]))
        # self.A = torch.Tensor(self.raw_data["A"])
        

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        yt = self.z[idx]
        xt = self.x[idx]
        ct = self.c[idx]
        idx_rnd = random.randint(0, len(self.z)-1)
        ytr = self.z[idx_rnd]
        xtr = self.x[idx_rnd]
        ctr = self.c[idx_rnd]
        sample = {"s1": {"yt": yt, "xt": xt, "ct": ct},
                  "s2": {"yt": ytr, "xt": xtr, "ct": ctr}
                  }
        return sample

class SimulationDatasetTSTwoSamplePCLNS(SimulationDatasetTSTwoSampleNS):
    def __init__(self, data_path, lags=2,c=False):
        super().__init__(data_path, c)
        self.L = lags
    def __getitem__(self, idx):
        yt = self.z[idx]
        xt = self.x[idx]
        ct = self.c[idx]
        xt_cur, xt_his = self.seq_to_pairs(xt)
        idx_rnd = random.randint(0, len(self.z)-1)
        ytr = self.z[idx_rnd]
        xtr = self.x[idx_rnd]
        ctr = self.c[idx_rnd]
        xtr_cur, xtr_his = self.seq_to_pairs(xtr)
        xt_cat = torch.cat((xt_cur, xt_his), dim=1)
        xtr_cat = torch.cat((xt_cur, xtr_his), dim=1)

        sample = {"s1": {"yt": yt, "xt": xt, "ct": ct},
                  "s2": {"yt": ytr, "xt": xtr, "ct": ctr},
                  "pos": {"x": xt_cat, "y": 1},
                  "neg": {"x": xtr_cat, "y": 0}
                  }
        return sample
    def seq_to_pairs(self, x):
        x = x.unfold(dimension = 0, size = self.L+1, step = 1)
        x = torch.swapaxes(x, 1, 2)
        xx, yy = x[:,-1:], x[:,:-1]
        return xx, yy

if __name__ == "__main__":
    data_path = "data/simulation/ls8_nc5_lags1/arhmm_pnl_change_gaussian_ts"
    dataset = ARHMNLICADataset(data_path)
    for batch in DataLoader(dataset, batch_size=10):
        print(batch)
        break