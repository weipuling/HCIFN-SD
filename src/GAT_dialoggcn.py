import torch
from torch import nn
import torch.nn.functional as F


def mask_logic(alpha, adj):
    '''
    performing mask logic with adj
    :param alpha:
    :param adj:
    :return:
    '''
    return alpha - (1 - adj) * 1e30


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    图注意力层
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  # 节点表示向量的输入特征维度
        self.out_features = out_features  # 节点表示向量的输出特征维度
        self.dropout = dropout  # dropout参数
        self.alpha = alpha  # leakyrelu激活的参数
        self.concat = concat  # 如果为true, 再进行elu激活

        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # xavier初始化

        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        """
        inp: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵 维度[N, N] 非零即一，数据结构基本知识
        """
        h = torch.mm(inp, self.W)  # [N, out_features]
        N = h.size()[0]  # N 图的节点数

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        # [N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # [N, N, 1] => [N, N] 图注意力的相关系数（未归一化）

        zero_vec = -1e12 * torch.ones_like(e)  # 将没有连接的边置为负无穷
        attention = torch.where(adj > 0, e, zero_vec)  # [N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=1)  # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, alpha, n_heads):
        """Dense version of GAT
        n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
        从不同的子空间进行抽取特征。
        """
        super(GAT, self).__init__()
        self.dropout = dropout

        # 定义multi-head的图注意力层
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  # 将每个head得到的表示进行拼接
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = F.elu(self.out_att(x, adj))  # 输出并激活
        return F.log_softmax(x, dim=1)  # log_softmax速度变快，保持数值稳定


class GAT_dialoggcn_v1(nn.Module):
    '''
    use linear to avoid OOM
    H_i = alpha_ij(W_rH_j),equation 5 in paper
    alpha_ij = attention(H_i, H_j),equation 4 in paper
    '''
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size * 2, 1)
        self.Wr0 = nn.Linear(hidden_size, hidden_size, bias = False)
        self.Wr1 = nn.Linear(hidden_size, hidden_size, bias = False)

    def forward(self, Q, K, V, adj, s_mask):
        '''
        imformation gatherer with linear attention
        :param Q: (B, D) # query utterance,(32,200)
        :param K: (B, N, D) # context,(32,78,200)
        :param V: (B, N, D) # context,(32,78,200)
        :param adj: (B,  N) # the adj matrix of the i th node
        :param s_mask: (B,  N) #
        :return:
        '''
        B = K.size()[0]    #B是batch size
        N = K.size()[1]    #N在这里是目前更新到的utterance下标l
        # print('Q',Q.size())
        Q = Q.unsqueeze(1).expand(-1, N, -1) # (16,l,200),把当前utterance特征Q升维到和K一个维度
        # print('K',K.size())
        X = torch.cat((Q,K), dim = 2) # (16,l,400)，拼接当前utterance和邻居节点特征,对应公式分子
        # print('X',X.size())
        alpha = self.linear(X).permute(0,2,1) #(B, 1, N),(16,1,l)
        #alpha = F.leaky_relu(alpha)
        # print('alpha',alpha.size())
        # print(alpha)
        adj = adj.unsqueeze(1)  # (B, 1, N)，(32,1,l)这里传入的是当前utterance的边关系下标，
        alpha = mask_logic(alpha, adj) # (B, 1, N),有边的下标不变，没边的下标为负无穷大，方便后续softmax
        #mask_logic:alpha - (1 - adj) * 1e30
        # print('alpha after mask',alpha.size())
        # print(alpha)

        attn_weight = F.softmax(alpha, dim = 2) # (16, 1, l)
        # print('attn_weight',attn_weight.size())
        # print(attn_weight)

        V0 = self.Wr0(V) # (16, l, 200)
        V1 = self.Wr1(V) # (16, l, 200)

        s_mask = s_mask.unsqueeze(2).float()   # (16, l, 1)   s_mask是当前utterance之前的所有utterance的说话人信息,如果是相同说话人则为1，否则为0
        V = V0 * s_mask + V1 * (1 - s_mask)#V(16, l, 200)

        # threshold = 0.2
        # temp_weight = attn_weight.clone()
        # temp_weight[temp_weight != 0] = 1  # 将非0元素置为1
        # non_zero_count = temp_weight.sum(dim=2)  # 计算非0元素的个数
        # avg_weight_test1 = torch.sum(attn_weight, dim=2)  # 计算非0元素的和
        # threshold = torch.where(non_zero_count > 0, avg_weight_test1 / non_zero_count, torch.tensor(0.).cuda())
        # # avg_weight = torch.masked_select(attn_weight,attn_weight!=0).mean()
        # updated_adj = torch.where(attn_weight >= threshold.unsqueeze(1), torch.tensor(1).cuda(), torch.tensor(0).cuda())

        attn_sum = torch.bmm(attn_weight, V).squeeze(1) # (16, 200),(16,1,l)*(16,l,200)
        # print('attn_sum',attn_sum.size())

        return attn_weight, attn_sum

class GAT_dialoggcn_v2(nn.Module):
    '''
    use linear to avoid OOM
    H_i = alpha_ij(W_rH_j)
    alpha_ij = attention(H_i, H_j, rel)
    '''
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size * 3, 1)
        self.Wr0 = nn.Linear(hidden_size, hidden_size, bias = False)
        self.Wr1 = nn.Linear(hidden_size, hidden_size, bias = False)
        self.rel_emb = nn.Embedding(2, hidden_size)

    def forward(self, Q, K, V, adj, s_mask):
        '''
        imformation gatherer with linear attention
        :param Q: (B, D) # query utterance
        :param K: (B, N, D) # context
        :param V: (B, N, D) # context
        :param adj: (B,  N) # the adj matrix of the i th node
        :param s_mask: (B,  N) #
        :return:
        '''
        rel_emb = self.rel_emb(s_mask) # (B, N, D)
        B = K.size()[0]
        N = K.size()[1]
        # print('Q',Q.size())
        Q = Q.unsqueeze(1).expand(-1, N, -1) # (B, N, D)；
        # print('K',K.size())
        X = torch.cat((Q,K,rel_emb), dim = 2) # (B, N, 3D)
        # print('X',X.size())
        alpha = self.linear(X).permute(0,2,1) #(B, 1, N)
        # print('alpha',alpha.size())
        # print(alpha)
        adj = adj.unsqueeze(1)
        alpha = mask_logic(alpha, adj) # (B, 1, N)
        # print('alpha after mask',alpha.size())
        # print(alpha)

        attn_weight = F.softmax(alpha, dim = 2) # (B, 1, N)
        # print('attn_weight',attn_weight.size())
        # print(attn_weight)

        V0 = self.Wr0(V) # (B, N,D)
        V1 = self.Wr1(V) # (B, N, D)

        s_mask = s_mask.unsqueeze(2).float()
        V = V0 * s_mask + V1 * (1 - s_mask)

        attn_sum = torch.bmm(attn_weight, V).squeeze(1) # (B, D)
        # print('attn_sum',attn_sum.size())

        return attn_weight, attn_sum

class attentive_node_features(nn.Module):
    '''
    Method to obtain attentive node features over the graph convoluted features
    '''

    def __init__(self, hidden_size):
        super().__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)

    def forward(self, features, lengths, nodal_att_type):
        '''
        features : (B, N, V)
        lengths : (B, )
        nodal_att_type : type of the final nodal attention
        '''

        if nodal_att_type == None:
            return features

        batch_size = features.size(0)
        max_seq_len = features.size(1)
        padding_mask = [l * [1] + (max_seq_len - l) * [0] for l in lengths]
        padding_mask = torch.tensor(padding_mask).to(features)  # (B, N)
        causal_mask = torch.ones(max_seq_len, max_seq_len).to(features)  # (N, N)
        causal_mask = torch.tril(causal_mask).unsqueeze(0)  # (1, N, N)

        if nodal_att_type == 'global':
            mask = padding_mask.unsqueeze(1)
        elif nodal_att_type == 'past':
            mask = padding_mask.unsqueeze(1) * causal_mask

        x = self.transform(features)  # (B, N, V)
        temp = torch.bmm(x, features.permute(0, 2, 1))
        # print(temp)
        alpha = F.softmax(torch.tanh(temp), dim=2)  # (B, N, N)
        alpha_masked = alpha * mask  # 用注意力权重来乘以mask(B, N, N)

        alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)  # (B, N, 1)
        # print(alpha_sum)
        alpha = alpha_masked / alpha_sum  # (B, N, N)
        attn_pool = torch.bmm(alpha, features)  # (B, N, V)

        return attn_pool