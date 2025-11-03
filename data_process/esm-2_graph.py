import pandas as pd
import torch
import esm
import math
import numpy as np
import matplotlib.pyplot as plt
import json, pickle
from collections import OrderedDict
import os
from tqdm import tqdm


def protein_graph_construct(df_proteins, save_dir):
    # Load ESM-2 model
    model_path = 'E:\MTE-DTA\大模型\esm2\esm2_t33_650M_UR50D.pt'  # 修改为你的 ESM-2 模型路径
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    target_graph = {}  # 存储目标图
    key_list = []  # 存储蛋白质ID的列表
    prots = []

    # 将蛋白质ID添加到list中
    for index, row in df_proteins.iterrows():
        prots.append(row['Sequence'])
        key_list.append(row['UniProtID'])

    # 使用tqdm显示进度条
    for k_i in tqdm(range(len(key_list))):
        key = key_list[k_i]
        data = []
        pro_id = key  # 蛋白质ID
        seq = prots[k_i]  # 蛋白质序列
        if len(seq) <= 300:
            data.append((pro_id, seq))
            batch_labels, batch_strs, batch_tokens = batch_converter(data)  # 转换为模型输入格式
            with torch.no_grad():  # 使用模型预测
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            contact_map = results["contacts"][0].numpy()  # 获取接触图
            target_graph[pro_id] = contact_map
        else:
            contact_prob_map = np.zeros((len(seq), len(seq)))  # global contact map prediction
            interval = 150  # 子序列间隔
            i = math.ceil(len(seq) / interval)  # 计算子序列的数量

            for s in range(i):  # 子序列预测接触图
                start = s * interval  # sub seq predict start
                end = min((s + 2) * interval, len(seq))  # sub seq predict end
                sub_seq_len = end - start
                # prediction 预测子序列接触图
                temp_seq = seq[start:end]
                temp_data = []
                temp_data.append((pro_id, temp_seq))
                batch_labels, batch_strs, batch_tokens = batch_converter(temp_data)
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33], return_contacts=True)

                # insert into the global contact map 插入全局接触图中
                row, col = np.where(contact_prob_map[start:end, start:end] != 0)
                row = row + start
                col = col + start
                contact_prob_map[start:end, start:end] = contact_prob_map[start:end, start:end] + results["contacts"][0].numpy()
                contact_prob_map[row, col] = contact_prob_map[row, col] / 2.0  # 平均化重叠区域
                if end == len(seq):
                    break
            # 将最终的接触图存储到目标图中
            target_graph[pro_id] = contact_prob_map

        # 保存接触图到指定目录
        np.save(save_dir + pro_id + '.npy', target_graph[pro_id])

if __name__ == '__main__':
    def save_obj(obj, name):  # 保存对象文件
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(name):  # 从文件中加载对象
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)

    # for test
    dataset = 'davis'
    df_proteins = pd.read_csv('E:\处理数据\提取好的数据\\target_IC50.csv',encoding='gbk')
    print('dataset:', dataset)
    print(len(df_proteins))

    save_dir = 'E:\MTE-DTA\data\IC50\sequence_representations_davis\\'
    if not os.path.exists(save_dir):  # 如果保存路径不存在，则创建目录
        os.makedirs(save_dir)
    protein_graph_construct(df_proteins, save_dir)  # 构建蛋白质图并保存

