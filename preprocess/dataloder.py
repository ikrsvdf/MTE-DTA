import ast
import os
import torch
import numpy as np
import pandas as pd
import networkx as nx
from rdkit import Chem
from tqdm import tqdm
from .utils import DTADataset
import warnings
from rdkit import RDLogger
import warnings
warnings.filterwarnings("ignore")
# 在代码开头添加，切换到项目根目录
project_root = 'MTE-DTA'  # 根据你的实际路径调整
os.chdir(project_root)
# 方法1：抑制所有RDKit警告
RDLogger.DisableLog('rdApp.*')

# 方法2：只抑制弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

def dic_normalize(dic):  # 对字典中的值进行归一化
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


# 定义配体的符号字典
VOCAB_LIGAND_ISO = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                    "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                    "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                    "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                    "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                    "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                    "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                    "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

# 定义氨基酸残基表及其符号字典
pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']  # 21个残基

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}  # 25个氨基酸

seq_dict = {v: (i + 1) for i, v in enumerate(CHARPROTSET)}  # 蛋白质序列数值化

max_seq_len = 1200


def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x


max_smi_len = 100


def smiles_cat(drug):
    x = np.zeros(max_smi_len)
    for i, ch in enumerate(drug[:max_smi_len]):
        x[i] = VOCAB_LIGAND_ISO[ch]
    return x


# 定义不同类别的氨基酸残基
pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

# 定义氨基酸残基的重量表
res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}
# 未未知残基（'X'）计算平均分子量
res_weight_table['X'] = np.average([res_weight_table[k] for k in res_weight_table.keys()])

# 定义氨基酸残基的pKa值表
res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}
res_pka_table['X'] = np.average([res_pka_table[k] for k in res_pka_table.keys()])

# 定义氨基酸残基的pKb值表
res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}
res_pkb_table['X'] = np.average([res_pkb_table[k] for k in res_pkb_table.keys()])

# 定义氨基酸残基的pKx值表
res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}
res_pkx_table['X'] = np.average([res_pkx_table[k] for k in res_pkx_table.keys()])

# 定义氨基酸残基的pL值表
res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}
res_pl_table['X'] = np.average([res_pl_table[k] for k in res_pl_table.keys()])

# 定义氨基酸残基在ph=2时的疏水性
res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}
res_hydrophobic_ph2_table['X'] = np.average([res_hydrophobic_ph2_table[k] for k in res_hydrophobic_ph2_table.keys()])

# 定义氨基酸残基在ph=7时的疏水性
res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}
res_hydrophobic_ph7_table['X'] = np.average([res_hydrophobic_ph7_table[k] for k in res_hydrophobic_ph7_table.keys()])

# nomarlize the residue feature  对各位属性表进行归一化处理
res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


# 对输入进行one-hot编码，如果输入不在允许集合中，则抛出异常
def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


# 对带有未知符号的输入进行one-hot编码，如果不在允许集合中，则使用最后一个值
# one ont encoding with unknown symbol
def one_hot_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# 对输入进行编码，处理未知符号
def encoding_unk(x, allowable_set):
    list = [False for i in range(len(allowable_set))]
    i = 0
    for atom in x:
        if atom in allowable_set:
            list[allowable_set.index(atom)] = True
            i += 1
    if i != len(x):
        list[-1] = True
    return list


# 获取氨基酸残基的特征
def seq_feature(seq):
    residue_feature = []
    for residue in seq:
        if residue not in pro_res_table:
            residue = 'X'  # 如果残基在这五个类别中则为1，否则为0  dim = 5
        res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                         1 if residue in pro_res_polar_neutral_table else 0,
                         1 if residue in pro_res_acidic_charged_table else 0,
                         1 if residue in pro_res_basic_charged_table else 0]
        res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue],
                         res_pkx_table[residue],  # 残基属性  dim = 7
                         res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
        residue_feature.append(res_property1 + res_property2)  # 每个残基 dim = 12

    pro_hot = np.zeros((len(seq), len(pro_res_table)))
    pro_property = np.zeros((len(seq), 12))
    for i in range(len(seq)):
        pro_hot[i,] = one_hot_encoding_unk(seq[i], pro_res_table)  # 残基 one-hot编码 dim=21
        pro_property[i,] = residue_feature[i]  # 残基属性 dim=12
    seq_feature = np.concatenate((pro_hot, pro_property), axis=1)
    return seq_feature  # 残基 dim=33


# 获取药物原子特征  每个原子节点 dim=78
def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_hot_encoding_unk(atom.GetSymbol(),
                                         ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                          'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                          'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                          'Pt', 'Hg', 'Pb', 'X']) +
                    one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_hot_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])



def bond_features(bond):  # 边的维度是17维度
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, 0, 0]
    bond_feats = bond_feats + one_hot_encoding_unk(bond.GetIsConjugated(), [0, 1, "nonbond"]) + \
                 one_hot_encoding_unk(bond.IsInRing(), [0, 1, "nonbond"])
    bond_feats = bond_feats + one_hot_encoding_unk(
        str(bond.GetStereo()),
        ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "nonbond"])
    results = np.array(bond_feats, dtype=np.float32)
    # results = np.array(bond_feats).astype(np.float32)
    return results


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    mol_size = mol.GetNumAtoms()
    mol_features = []
    for atom in mol.GetAtoms():
        feats = atom_features(atom)
        mol_features.append(feats / sum(feats))
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = bond_features(bond)
        # 添加双向边
        edge_index += [[i, j], [j, i]]
        edge_attr += [feat, feat]
    # 添加自环（可选）
    for idx in range(mol_size):
        edge_index.append([idx, idx])
        edge_attr.append([0] * 17)
    edge_index = np.array(edge_index)
    return mol_size, mol_features, edge_index, edge_attr


# 构建蛋白质残基图
def sequence_to_graph(target_key, target_sequence, distance_dir):
    target_edge_index = []  # target_key ：蛋白质ID
    target_edge_distance = []
    target_size = len(target_sequence)  # 蛋白质长度

    contact_map_file = os.path.join(distance_dir, target_key + '.npy')
    distance_map = np.load(contact_map_file)  # 获取蛋白质接触图

    for i in range(target_size):
        distance_map[i, i] = 1  # 残基自身距离为 1
        if i + 1 < target_size:
            distance_map[i, i + 1] = 1  # 相邻残基之间距离也为 1
    index_row, index_col = np.where(distance_map >= 0.5)  # for threshold # 获取哪些残基之间距离>=0.5

    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])  # dege # 大于0.5的边添加到索引表中
        target_edge_distance.append(distance_map[i, j])  # edge weight # 边的距离添加

    target_feature = seq_feature(target_sequence)  # 获取残基节点特征

    return target_size, target_feature, target_edge_index, target_edge_distance  # 返回序列长度，节点特征，节点边索引，节点边距离


def create_DTA_dataset(dataset_name, type, dataset, force_reprocess=False):
    # 加载蛋白质特征和距离图
    process_dir = os.path.join('data')
    pro_distance_dir = os.path.join(process_dir, dataset_name, 'sequence_representations_davis')

    # 加载药物数据
    ligands_paths = os.path.join('data', dataset_name, 'drug_IC50.csv')
    df_ligands = pd.read_csv(ligands_paths)

    # 加载蛋白质数据
    proteins_paths = os.path.join('data', dataset_name, 'target_IC50.csv')
    df_proteins = pd.read_csv(proteins_paths)

    # 构建药物图字典
    smile_graph = {}
    for index, row in tqdm(df_ligands.iterrows(), total=len(df_ligands)):
        smile = row['Ligand SMILES']
        smile_graph[smile] = smile_to_graph(smile)

    # 构建蛋白质图字典
    target_graph = {}
    for index, row in tqdm(df_proteins.iterrows(), total=len(df_proteins)):
        prot_key = row['UniProtID']
        sequence = row['Sequence']
        target_graph[prot_key] = sequence_to_graph(prot_key, sequence, pro_distance_dir)

    # 加载嵌入向量
    smiles_path = os.path.join('data', dataset_name, 'drug_embedding.npy')
    fasta_path = os.path.join('data', dataset_name, 'sequence_300_acid.npy')

    df_smiles_emd = np.load(smiles_path, allow_pickle=True).item()
    df_fasta_emd = np.load(fasta_path, allow_pickle=True).item()

    smiles_emb = {}
    for smile, emb in tqdm(df_smiles_emd.items(), total=len(df_smiles_emd)):
        smiles_emb[smile] = emb.astype(np.float32)

    target_emb = {}
    for prot_key, emb in tqdm(df_fasta_emd.items(), total=len(df_fasta_emd)):
        target_emb[prot_key] = emb.astype(np.float32)

    # 加载训练、验证、测试数据
    # train_csv = os.path.join('data', dataset_name, type, dataset, 'data_train_new.csv')
    train_csv = os.path.join('data', dataset_name, type, dataset, 'data_train.csv')
    val_csv = os.path.join('data', dataset_name, type, dataset, 'data_val.csv')
    test_csv = os.path.join('data', dataset_name, type, dataset, 'data_test.csv')

    df_train = pd.read_csv(train_csv,encoding='gbk')
    df_val = pd.read_csv(val_csv,encoding='gbk') 
    df_test = pd.read_csv(test_csv, encoding='gbk' )

    # 提取数据
    train_drugs, train_prots, train_target_key, train_Y = (
        list(df_train['Ligand SMILES']), list(df_train['Sequence']),
        list(df_train['UniProtID']), list(df_train['Average_pIC50'])
    )
    val_drugs, val_prots, val_target_key, val_Y = (
        list(df_val['Ligand SMILES']), list(df_val['Sequence']),
        list(df_val['UniProtID']), list(df_val['Average_pIC50'])
    )
    test_drugs, test_prots, test_target_key, test_Y = (
        list(df_test['Ligand SMILES']), list(df_test['Sequence']),
        list(df_test['UniProtID']), list(df_test['Average_pIC50'])
    )

    # 转换为numpy数组
    train_drugs, train_target_key, train_Y = (
        np.asarray(train_drugs), np.asarray(train_target_key), np.asarray(train_Y)
    )
    val_drugs, val_target_key, val_Y = (
        np.asarray(val_drugs), np.asarray(val_target_key), np.asarray(val_Y)
    )
    test_drugs, test_target_key, test_Y = (
        np.asarray(test_drugs), np.asarray(test_target_key), np.asarray(test_Y)
    )

    # 创建数据集实例
    train_data = DTADataset(
        dataset_name, train_drugs, train_Y, train_target_key,
        smiles_emb, target_emb, smile_graph, target_graph
    )
    val_data = DTADataset(
        dataset_name, val_drugs, val_Y, val_target_key,
        smiles_emb, target_emb, smile_graph, target_graph
    )
    test_data = DTADataset(
        dataset_name, test_drugs, test_Y, test_target_key,
        smiles_emb, target_emb, smile_graph, target_graph
    )

    return train_data, val_data, test_data