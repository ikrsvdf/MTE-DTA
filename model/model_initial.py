import copy
from inspect import isfunction
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gep, global_max_pool as gmp, global_add_pool as gap
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from .Transformer import Graph_Transformer


class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        # talking heads
        self.pre_softmax_talking_heads = nn.Conv2d(n_heads, n_heads, 1, bias=False)
        self.post_softmax_talking_heads = nn.Conv2d(n_heads, n_heads, 1, bias=False)
        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]

        # query = key = value [batch size, sent len, hid dim]
        Q = self.w_q(query)  # (64, 24, 128)
        K = self.w_k(key)  # (64, 24, 128)
        V = self.w_v(value)  # (64, 24, 128)

        # Q, K, V = [batch size, sent len, hid dim]
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)  # (64, 8, 24, 16)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)  # (64, 8, 24, 16)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)  # (64, 8, 24, 16)

        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale  # (64, 8, 24, 24)

        energy = self.pre_softmax_talking_heads(energy)  # (64, 8, 24, 24)
        # energy = [batch size, n heads, sent len_Q, sent len_K]
        # if mask is not None:
        #     mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1]
        #     mask = mask.expand(-1, seq_len_Q, seq_len_K)  # [batch_size, seq_len_Q, seq_len_K]
        #     energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))
        attention = self.post_softmax_talking_heads(attention)  # (64, 8, 24, 24)
        # attention = [batch size, n heads, sent len_Q, sent len_K]
        x = torch.matmul(attention, V)  # (64, 8, 24, 16)

        # x = [batch size, n heads, sent len_Q, hid dim // n heads]
        x = x.permute(0, 2, 1, 3).contiguous()  # (64, 24, 8, 16)

        # x = [batch size, sent len_Q, n heads, hid dim // n heads]
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))  # (64, 24, 128)

        # x = [batch size, src sent len_Q, hid dim]
        x = self.fc(x)  # (64, 24, 128)

        # x = [batch size, sent len_Q, hid dim]
        return x


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GLU(nn.Module):
    def __init__(self, dim_in, dim_out, activation):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate)


def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)


class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, swish=False, relu_squared=False,
                 post_act_ln=False, dropout=0., no_bias=False, zero_init_output=False):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if relu_squared:
            activation = ReluSquared()
        elif swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()  #

        project_in = nn.Sequential(  # 前馈神经网络，用到的是GEL
            nn.Linear(dim, inner_dim, bias=not no_bias),
            activation
        ) if not glu else GLU(dim, inner_dim, activation)

        self.ff = nn.Sequential(
            project_in,
            nn.LayerNorm(inner_dim) if post_act_ln else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias=not no_bias)
        )

        # init last linear layer to 0
        if zero_init_output:
            init_zero_(self.ff[-1])

    def forward(self, x):
        return self.ff(x)


def get_nlayers(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# 编码器用于序列编码， (CNN+GLU激活）+残差连接
class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, kernel_size, dropout, device):
        super().__init__()
        # 确保卷积核大小为奇数
        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)  # 缩放因子，用于残差连接的缩放
        self.conv = nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size,
                              padding=(kernel_size - 1) // 2)  # for _ in range(self.n_layers)])  # convolutional layers
        self.dropout = nn.Dropout(dropout)

    def forward(self, conv_input):
        # protein = [batch size, protein len,protein_dim]
        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)
        # conv_input = [batch size, hid dim, protein len] (64, 128, 300)
        # pass through convolutional layer
        conved = self.conv(self.dropout(conv_input))
        # conved = [batch size, 2*hid dim, protein len] (64, 256, 300)
        # pass through GLU activation function
        conved = F.glu(conved, dim=1)  # 将glu改成relu
        # conved = [batch size, hid dim, protein len]  (64, 128, 300)
        # apply residual connection / high way
        conved = (conved + conv_input) * self.scale
        # conved = [batch size, hid dim, protein len]  (64, 128, 300)
        # set conv_input to conved for next loop iteration
        conved = conved.permute(0, 2, 1)
        # conved = [batch size,protein len,hid dim]  (64, 300, 128)
        return conved


### 解码器，就是transformer架构
class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.sa = SelfAttention(hid_dim, n_heads, dropout, device)  # # x = [batch size, sent len_Q, hid dim]
        self.ea = SelfAttention(hid_dim, n_heads, dropout, device)
        self.ff = FeedForward(hid_dim, hid_dim, glu=True, dropout=dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # self attetion 自注意力：首先通过层归一化将目标序列（trg）和自注意力结果相加，然后进行Dropout，最后通过层归一化
        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))  # (64, 24, 128)
        # cross attention 将目标序列（trg）和源序列（src）计算交叉注意力，结果加到目标序列上，之后进行层归一化和Dropout
        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))  # (64, 24, 128)
        # feed forward 前馈神经网络：将目标序列传入前馈网络，结果加到目标序列上，再进行层归一化和Dropout
        trg = self.ln(trg + self.do(self.ff(trg)))  # (64, 24, 128)
        return trg  # # x = [batch size, sent len_Q, hid dim]


### 2D卷积块
def Multilevelblock(in_layers, out_layer):
    blk = nn.Conv2d(in_layers, out_layer, 1, bias=False)
    return blk


class Highway(nn.Module):
    r"""Highway Layers
    Args:
        - num_highway_layers(int): number of highway layers.
        - input_size(int): size of highway input.
    """

    def __init__(self, num_highway_layers, input_size):
        super(Highway, self).__init__()
        self.num_highway_layers = num_highway_layers
        self.non_linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.gate = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        for layer in range(self.num_highway_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            # Compute percentage of non linear information to be allowed for each element in x
            non_linear = F.relu(self.non_linear[layer](x))
            # Compute non linear information
            linear = self.linear[layer](x)
            # Compute linear information
            x = gate * non_linear + (1 - gate) * linear
            # Combine non linear and linear information according to gate
            x = self.dropout(x)
        return x


class DualInteract(nn.Module):

    def __init__(self, field_dim, embed_size, head_num, dropout=0.2):
        super(DualInteract, self).__init__()
        self.bit_wise_net = Highway(input_size=field_dim * embed_size,
                                    num_highway_layers=2)  # 原來是2，改成3了

    def forward(self, x):
        """
            x : batch, field_dim, embed_dim
        """
        b, f, e = x.shape
        bit_wise_x = self.bit_wise_net(x.reshape(b, f * e))
        m_x = bit_wise_x
        return m_x


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(Attention, self).__init__()
        self.project_x = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.project_xt = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.project_xq = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, x, xt):
        x = self.project_x(x)
        xt = self.project_xt(xt)
        a = torch.cat((x, xt), 1)
        a = torch.softmax(a, dim=1)
        return a


from torch_geometric.nn import TransformerConv, global_max_pool as gmp


class HierarchyT(nn.Module):
    def __init__(self, protein_dim, drug_dim, hid_dim, atom_mid_dim, num_features_mol, num_features_pro, n_heads,
                 n_enlayers, n_delayers, device):
        # protein_dim = 1280, hid_dim = 128, atom_dim = 32, n_heads = 8, n_enlayers = 2, n_delayers = 2, dropout = 0.2
        super().__init__()
        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.n_enlayers = n_enlayers  # 编码器层数、多级注意力层数
        self.n_heads = n_heads
        self.bn = nn.BatchNorm1d(atom_mid_dim)  # 这里应该是处理的药物
        self.n_delayers = n_delayers  # 解码器层数
        self.dropout = 0.1
        self.garph_transformer = Graph_Transformer(drug_dim, 17, self.hid_dim, [64, 64],
                                                   [2, 2], 64, 0.1)

        self.pro_conv1 = SAGEConv(num_features_pro, num_features_pro, aggr='sum')  # 刚刚改成mean了
        self.pro_conv2 = SAGEConv(num_features_pro, num_features_pro, aggr='sum')
        self.pro_conv3 = SAGEConv(num_features_pro, num_features_pro, aggr='sum')
        self.pro_fc1 = nn.Linear(in_features=num_features_pro, out_features=128)
        self.bn1 = nn.BatchNorm1d(num_features_pro)
        self.bn2 = nn.BatchNorm1d(num_features_pro)
        self.bn3 = nn.BatchNorm1d(num_features_pro)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(0.1)

        self.Encoder = nn.ModuleList(  # 特征嵌入（CNN+GLU+残差）
            [EncoderLayer(hid_dim, i * 2 + 3, self.dropout, device) for i in range(n_enlayers)])

        self.Decoder = nn.ModuleList(  # 特征交互 （transformer）
            [DecoderLayer(hid_dim, n_heads, self.dropout, device) for i in range(n_delayers)])

        Multi_level_att_block = []  # 2DCNN
        for i in range(n_enlayers):
            Multi_level_att_block.append(Multilevelblock(1 + i, 1))
        self.multilevelattention = nn.ModuleList(Multi_level_att_block)

        self.device = device  # 线性注意力
        self.protein_attention = nn.Linear(hid_dim, hid_dim)
        self.compound_attention = nn.Linear(hid_dim, hid_dim)
        self.inter_attention = nn.Linear(hid_dim, hid_dim)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()
        proj_dim = 128
        self.feature_interact = DualInteract(field_dim=4, embed_size=proj_dim, head_num=8)  # 修改处 dim=3改成5
        hidden_dim = 256
        self.attention = Attention(hidden_dim)
        # 对比
        # self.constrast = SupConLoss()

        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.ft = nn.Linear(atom_mid_dim, hid_dim)
        self.fc1 = nn.Linear(hid_dim * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)

        self.protmaxpool = nn.AdaptiveMaxPool1d(1)
        self.drugmaxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, proteins, trgs, trg_mask=None, src_mask=None):
        ## 1）局部特征提取
        protein = proteins.fasta_emb
        batch_size = protein.shape[0]  # 获取批次大小
        # batch_size = protein.fasta_emb
        trg = trgs.smiles_emb
        trg = trg.reshape(batch_size, -1, 32)  ## 药物   (64,768) --> (64,24,32),数据是tensor类型，这里没有batch，就是一个样本
        src = self.fc(protein)  # 蛋白质  (64,300,1280)--> (64,300,128)
        trg = self.ft(trg)  # 药物   (64,24,32) --> (64,24,128)
        # Multilevel att & talking heads

        # 初始化talkingsrc为零张量，用于存储不同Encoder层的输出
        talkingsrc = torch.zeros([batch_size, 1, src.shape[1], self.hid_dim]).to(self.device)  # (64, 1, 300, 128)

        # 经过每一层Encoder进行编码
        for i in range(self.n_enlayers):
            src = self.Encoder[i](src)  # (64, 1, 300, 128)
            talkingsrc = torch.cat([talkingsrc, src.unsqueeze(1)],
                                   dim=1)  # (64, 2, 300, 128)-> # (64, 3, 300, 128)# 将每一层的编码输出拼接到talkingsrc上，得到蛋白质的特征
            x = talkingsrc[:, 1:, :, :]  # (64, 1, 300, 128)
            usesrc = self.multilevelattention[i](x).squeeze(dim=1)  # (64,300,128)# 多层注意力
            trg = self.Decoder[i](trg, usesrc, trg_mask,
                                  src_mask)  # (64,24,128) # 解码药物对蛋白质的交互矩阵，药物-蛋白质使用transformer,相当于将蛋白质特征与药物特征进行对齐

        # classifier
        src_att = self.protein_attention(talkingsrc[:, -1])  # (64, 300, 128) 取最后一层的特征# 使用线性注意力机制，计算蛋白质的注意力分数和药物的注意力分数
        trg_att = self.compound_attention(trg)  # (64, 24, 128)

        # 计算注意力矩阵
        # 计算注意力矩阵
        p_att = torch.unsqueeze(src_att, 1).repeat(1, trg.shape[1], 1, 1)  # # (64, 24, 300, 128) 扩展蛋白质的注意力矩阵
        c_att = torch.unsqueeze(trg_att, 2).repeat(1, 1, src.shape[1],
                                                   1)  # # (64, 24, 300, 128)扩展药物的注意力矩阵        # # 交互注意力
        Attention_matrix = self.inter_attention(self.relu(c_att + p_att))  # (64, 24, 300, 128)

        # 平均池化得到Compound和Protein对dT交互的注意力
        Compound_attetion = torch.mean(Attention_matrix, 2)  # 得到药物对交互特征的注意力分数 (64, 24, 128)
        Protein_attetion = torch.mean(Attention_matrix, 1)  # 得到蛋白质对交互特征的注意力分数 (64, 300, 128)
        Compound_attetion = self.sigmoid(Compound_attetion.permute(0, 2, 1))  # (64, 128, 24)
        Protein_attetion = self.sigmoid(Protein_attetion.permute(0, 2, 1))  # (64, 128, 300)

        # 利用注意力矩阵调整药物特征和蛋白质特征
        CompoundConv = trg.permute(0, 2, 1) * 0.5 + trg.permute(0, 2, 1) * Compound_attetion  # (64, 128, 24)
        # ProteinConv = talkingsrc[:, -1].unsqueeze(dim=0).permute(0, 2, 1) * 0.5 + talkingsrc[:, -1].unsqueeze(dim=0).permute(0, 2, 1) * Protein_attetion  # (1, 128, 300)
        ProteinConv = talkingsrc[:, -1].permute(0, 2, 1) * 0.5 + talkingsrc[:, -1].permute(0, 2,
                                                                                           1) * Protein_attetion  # (64, 128, 300)

        # 进行池化操作
        CompoundConv = self.drugmaxpool(CompoundConv).squeeze(2)  # (64, 128)
        ProteinConv = self.protmaxpool(ProteinConv).squeeze(2)  # (64, 128)
        pairS = torch.cat([CompoundConv, ProteinConv], dim=1)  # (64, 256)

        ## 2）全局特征提取protein,trg
        # mol_x, mol_edge_index, mol_edge_attr, mol_batch = trgs.x, trgs.edge_index, trgs.edge_attr, trgs.batch
        target_x, target_edge_index, target_batch = proteins.x, proteins.edge_index, proteins.batch

        # 获取小分子和蛋白质输入的结构信息
        molGraph = self.garph_transformer(trgs)

        # 第1层 GNN + BN + ReLU + Dropout
        x1 = self.pro_conv1(target_x, target_edge_index)
        # x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.dropout_layer(x1)

        # 第2层 GNN + 残差 + BN + ReLU + Dropout
        x2 = self.pro_conv2(x1, target_edge_index)
        # x2 = self.bn2(x2)
        x2 = self.relu(x2 + x1)
        x2 = self.dropout_layer(x2)

        # 第3层 GNN + 残差 + BN + ReLU + Dropout
        x3 = self.pro_conv3(x2, target_edge_index)
        # x3 = self.bn3(x3)
        x3 = self.relu(x3 + x2)
        x3 = self.dropout_layer(x3)

        # 全局池化（如 mean/max/sum）
        tarGraph = gep(x3, target_batch)  # (batch, num_features_pro)

        # 最后线性映射到 128 维，接 BN 和 ReLU
        tarGraph = self.pro_fc1(tarGraph)  # (batch, 128)
        tarGraph = self.relu(tarGraph)
        tarGraph = self.dropout_layer(tarGraph)

        pairG = torch.cat([molGraph, tarGraph], dim=1)  # (64, 256)
        # emb = torch.cat([pairS, pairG], dim=1)  # (64, 256)

        # 3） 对比学习+特征融合 在这里加一个局部DT特征与全局DT特征的对比学习以及门控机制融合两个特征，最终得到输出的特征
        a = self.attention(pairS, pairG)
        emb = torch.stack([pairS, pairG], dim=1)
        a = a.unsqueeze(dim=2)
        emb = (a * emb).reshape(-1, 2 * 256)

        # add some dense layers
        # 分类
        fully1 = self.leaky_relu(self.fc1(emb))  # (64, 1024)
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout2(fully2)
        label = self.out(fully2)  # (64, 1)

        return label  # 改成一个值
