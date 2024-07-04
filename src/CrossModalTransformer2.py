import numpy as np
import torch
import transformers.modeling_transfo_xl
from torch import nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):#attn_mask(78,n_head,32,1)
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        if attn_mask is not None:
            scores.masked_fill_(attn_mask == 0, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_Q = nn.Linear(d_model, self.d_k * n_heads)
        self.W_K = nn.Linear(d_model, self.d_k * n_heads)
        self.W_V = nn.Linear(d_model, self.d_k * n_heads)
        self.linear = nn.Linear(n_heads * self.d_k, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):#attn_mask(32,78,78),Q(32,78,200)
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.conv1 = nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self,d_model,n_heads):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.enc_self_attn = MultiHeadAttention(self.d_model,self.n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model)

    def forward(self, query,key,mask):
        enc_outputs, attn = self.enc_self_attn(query,key,key,mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self,d_model,n_heads,n_layers):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([EncoderLayer(d_model,n_heads) for _ in range(self.n_layers)])

    def forward(self, query,key,mask): # enc_inputs : [batch_size x source_len]
        enc_outputs = query
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs,key,mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class CrossModalTransformer2(nn.Module):
    def __init__(self,d_model,n_heads,n_layers):
        super(CrossModalTransformer2, self).__init__()
        self.encoder = Encoder(d_model,n_heads,n_layers)
    def forward(self, query,key,mask):
        enc_outputs, enc_self_attns = self.encoder(query,key,mask)
        return enc_outputs,enc_self_attns

class SelfMultiHeadAttention(nn.Module):
    '''只做多头注意力分数的计算，不做其他任何操作'''
    def __init__(self, d_model, n_heads):
        super(SelfMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_Q = nn.Linear(d_model, self.d_k * n_heads)
        self.W_K = nn.Linear(d_model, self.d_k * n_heads)
        self.W_V = nn.Linear(d_model, self.d_k * n_heads)
        self.linear = nn.Linear(n_heads * self.d_k, d_model)

    def forward(self, query, key, value, mask=None):
        q_s = self.W_Q(query).view(query.size(0), -1, self.n_heads, self.d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(key).view(query.size(0), -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(value).view(query.size(0), -1, self.n_heads, self.d_k).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]
        d_k = q_s.size(-1)
        scores = torch.matmul(q_s, k_s.transpose(-1, -2)) / np.sqrt(d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, v_s)
        context = context.transpose(1, 2).contiguous().view(context.size(0), -1,self.n_heads * self.d_k)  # context: [batch_size x len_q x n_heads * d_v]
        return context