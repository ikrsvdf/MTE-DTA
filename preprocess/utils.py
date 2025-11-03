import os

import pandas as pd
import os

import pandas as pd
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
import torch
from tqdm import tqdm
import numpy as np
from math import sqrt
from scipy import stats
from sklearn.metrics import r2_score
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")

# initialize the dataset
class DTADataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, drugs, labels, target_keys, smiles_emb, target_emb, smile_graph, target_graph,
                 root='data', transform=None, pre_transform=None):
        self.dataset_name = dataset_name
        self.drugs = drugs
        self.labels = labels
        self.target_keys = target_keys
        self.smiles_emb = smiles_emb
        self.target_emb = target_emb
        self.smile_graph = smile_graph
        self.target_graph = target_graph
        self.transform = transform
        self.pre_transform = pre_transform

    def __len__(self):
        return len(self.drugs)

    def __getitem__(self, idx):
        smile = self.drugs[idx]
        label = self.labels[idx]
        target_key = self.target_keys[idx]

        # 获取图结构信息
        mol_size, mol_features, mol_edge_index, edge_attr = self.smile_graph[smile]
        target_size, target_features, target_edge_index, target_edge_weight = self.target_graph[target_key]

        # 获取嵌入向量
        drug_emb = self.smiles_emb[smile]
        target_emb = self.target_emb[target_key]

        # 构造药物图数据
        GCNData_mol = DATA.Data(
            x=torch.Tensor(mol_features),
            edge_index=torch.LongTensor(mol_edge_index).transpose(1, 0),
            edge_attr=torch.FloatTensor(edge_attr),
            y=torch.FloatTensor([label])
        )
        GCNData_mol.smiles_emb = torch.Tensor([drug_emb])
        GCNData_mol.c_size = torch.LongTensor([mol_size])

        # 构造蛋白质图数据
        GCNData_pro = DATA.Data(
            x=torch.Tensor(target_features),
            edge_index=torch.LongTensor(target_edge_index).transpose(1, 0),
            edge_weight=torch.FloatTensor(target_edge_weight),
            y=torch.FloatTensor([label])
        )
        GCNData_pro.fasta_emb = torch.Tensor([target_emb])
        GCNData_pro.target_size = torch.LongTensor([target_size])

        # 应用预处理变换
        if self.pre_transform is not None:
            GCNData_mol = self.pre_transform(GCNData_mol)
            GCNData_pro = self.pre_transform(GCNData_pro)

        return GCNData_mol, GCNData_pro




def rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


def mse1(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs

def CI2(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

def ci(gt, pred):
    gt_mask = gt.reshape((1, -1)) > gt.reshape((-1, 1))
    diff = pred.reshape((1, -1)) - pred.reshape((-1, 1))
    h_one = (diff > 0)
    h_half = (diff == 0)
    CI = np.sum(gt_mask * h_one * 1.0 + gt_mask * h_half * 0.5) / np.sum(gt_mask)

    return CI

def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))

def collatef(batch):
    batchlist=[]
    for item in batch:
        compound,protein,interaction = item
        list=[compound,protein,interaction]
        batchlist.append(list)
    return batchlist