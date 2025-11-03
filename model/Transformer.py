import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
import copy
from torch_geometric.nn import (
    TransformerConv,
    SAGPooling,
    LayerNorm,
    global_add_pool,
    Linear,
)


# from mlayers import *

# self.model = MVN_DDI(78, 17, 128, 128, 0, [64, 64],
#                      [2, 2], 64, 0.0)
#
class CoAttentionLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features // 2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features // 2))
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features // 2))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))

    def forward(self, receiver, attendant):
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q
        values = receiver

        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        e_scores = torch.tanh(e_activations) @ self.a
        attentions = e_scores
        return attentions


class RESCAL(nn.Module):
    def __init__(self, n_rels, n_features):
        super().__init__()
        self.n_rels = n_rels
        self.n_features = n_features
        self.rel_emb = nn.Embedding(self.n_rels, n_features * n_features)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, heads, tails, rels, alpha_scores):
        rels = self.rel_emb(rels)

        rels = F.normalize(rels, dim=-1)
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)

        rels = rels.view(-1, self.n_features, self.n_features)

        scores = heads @ rels @ tails.transpose(-2, -1)

        if alpha_scores is not None:
            scores = alpha_scores * scores
        scores = scores.sum(dim=(-2, -1))

        return scores

    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_rels}, {self.rel_emb.weight.shape})"


class Graph_Transformer(nn.Module):
    def __init__(self, in_node_features, in_edge_features, hidd_dim, heads_out_feat_params,
                 blocks_params, edge_feature, dp):
        super().__init__()
        self.in_node_features = in_node_features
        self.in_edge_features = in_edge_features
        self.hidd_dim = hidd_dim
        self.n_blocks = len(blocks_params)

        self.initial_node_feature = Linear(self.in_node_features, self.hidd_dim, bias=True, weight_initializer='glorot')
        self.initial_edge_feature = Linear(self.in_edge_features, edge_feature, bias=True, weight_initializer='glorot')
        self.initial_node_norm = LayerNorm(self.hidd_dim)

        self.blocks = []
        for i, (head_out_feats, n_heads) in enumerate(zip(heads_out_feat_params, blocks_params)):
            block = Transformer(self.hidd_dim, n_heads, head_out_feats, edge_feature, dp)  # 128,2,64,64,0.1
            self.add_module(f"block{i}", block)
            self.blocks.append(block)

    def forward(self, h_data):  # h.x = [531,78],edge_index= [2,1703],edge_attr = [1703, 17]]
        h_data.x = self.initial_node_feature(h_data.x)  # h.x = [531,128],edge_index= [2,1703],edge_attr = [1703, 17]]
        h_data.x = self.initial_node_norm(h_data.x, h_data.batch)
        h_data.x = F.elu(h_data.x)
        h_data.edge_attr = self.initial_edge_feature(
            h_data.edge_attr)  # h.x = [531,128],edge_index= [2,1703],edge_attr = [1703, 64]]
        h_data.edge_attr = F.elu(h_data.edge_attr)

        repr_h = []
        for i, block in enumerate(self.blocks):
            h_global_graph_emb = block(h_data)  # [16,128]
            repr_h.append(h_global_graph_emb)

        h_data_fin = 0.6 * repr_h[0] + 0.4 * repr_h[-1]
        return h_data_fin


class Transformer(nn.Module):  # (128, 2, 64, 64, 0.1)
    def __init__(self, in_features, n_heads, head_out_feats, edge_feature, dropout_rare):
        super().__init__()
        # 初始化参数
        self.n_heads = n_heads  # 多头注意力的头数
        self.in_features = in_features  # 输入节点特征维度
        self.out_features = head_out_feats  # 每个头输出的特征维度

        # 第一层 TransformerConv（节点+边特征）
        self.feature_conv = TransformerConv(in_features, head_out_feats, n_heads, edge_dim=edge_feature,
                                            dropout=dropout_rare
                                            )
        # 边特征升维（用于第一层）
        self.lin_up = Linear(64, 64, bias=True, weight_initializer='glorot')

        # 第二层 TransformerConv
        self.feature_conv2 = TransformerConv(
            in_features, head_out_feats, n_heads,
            edge_dim=edge_feature, dropout=dropout_rare
        )
        # 第二层边特征升维
        self.lin_up2 = Linear(64, 64, bias=True, weight_initializer='glorot')

        # 自适应图池化，用于提取全局图表征
        self.readout = SAGPooling(n_heads * head_out_feats, min_score=-1)

        # 拼接边+节点后的重构层（未使用）
        self.re_shape = Linear(64 + 128, 128, bias=True, weight_initializer='glorot')

        # 层归一化（第1层）
        self.norm = LayerNorm(n_heads * head_out_feats)
        # 层归一化（第2层）
        self.norm2 = LayerNorm(n_heads * head_out_feats)

        # 边特征的图级表征升维（边信息增强）
        self.re_shape_e = Linear(64, 128, bias=True, weight_initializer='glorot')

    def forward(self, h_data):
        # 第1层 TransformerConv 更新节点和边特征
        h_data = self.ne_update(h_data)

        # 第2层 TransformerConv 更新节点特征
        h_data.x = self.feature_conv2(h_data.x, h_data.edge_index, h_data.edge_attr)

        # 边特征再次升维
        h_data.edge_attr = self.lin_up2(h_data.edge_attr)

        # 计算图级表示（节点+边）
        h_global_graph_emb = self.GlobalPool(h_data)

        # 归一化并激活（图级表征）
        h_global_graph_emb = nn.ELU()(F.normalize(h_global_graph_emb, 2, 1))

        # 第2层归一化与激活（节点特征）
        h_data.x = F.elu(self.norm2(h_data.x, h_data.batch))

        # 第二层边特征激活
        h_data.edge_attr = F.elu(h_data.edge_attr)

        return h_global_graph_emb

    def ne_update(self, h_data):
        # 第1层 TransformerConv（节点特征）
        h_data.x = self.feature_conv(h_data.x, h_data.edge_index, h_data.edge_attr)
        # 层归一化+激活
        h_data.x = F.elu(self.norm(h_data.x, h_data.batch))
        # 边特征升维+激活
        h_data.edge_attr = self.lin_up(h_data.edge_attr)
        h_data.edge_attr = F.elu(h_data.edge_attr)

        return h_data

    def GlobalPool(self, h_data):
        # 使用 SAGPooling 对节点进行自适应池化（h 和 t）
        h_att_x, att_edge_index, att_edge_attr, h_att_batch, att_perm, h_att_scores = self.readout(
            h_data.x, h_data.edge_index, edge_attr=h_data.edge_attr, batch=h_data.batch
        )
        # 通过池化后的节点特征做图级表示（global_add_pool）
        h_global_graph_emb = global_add_pool(h_att_x, h_att_batch)
        # 设置边特征作为新的 node.x 以便也做图级池化
        edge_batch = h_data.batch[h_data.edge_index[0]]
        h_global_graph_emb_edge = global_add_pool(h_data.edge_attr, edge_batch)
        # h_data_edge.x = h_data.edge_attr
        # h_global_graph_emb_edge = global_add_pool(h_data_edge.x, batch=h_data_edge.batch)
        # 边级表示升维后激活
        h_global_graph_emb_edge = F.elu(self.re_shape_e(h_global_graph_emb_edge))
        # 节点图表示 × 边图表示（融合节点和边信息）
        h_global_graph_emb = h_global_graph_emb * h_global_graph_emb_edge

        return h_global_graph_emb
