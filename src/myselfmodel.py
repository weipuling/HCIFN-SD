import copy
import math

import numpy as np
import torch
import transformers.modeling_ctrl
import transformers.modeling_transfo_xl
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
from CrossModalTransformer2 import MultiHeadAttention, PoswiseFeedForwardNet, \
    EncoderLayer, Encoder, CrossModalTransformer2, ScaledDotProductAttention, SelfMultiHeadAttention
from src.CS_GRU import CS_GRU_v1, GRUCell, CS_GRU_v2, GRUCell_v2, GRUCell_v3
from src.GAT_dialoggcn import GAT_dialoggcn_v1, attentive_node_features
from src.GAT_dialoggcn import GAT_dialoggcn_v2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SSE(nn.Module):
    def __init__(self, D_hidden):
        super().__init__()
        self.D_hidden = D_hidden
        self.init_trans = nn.Linear(self.D_hidden, self.D_hidden)
        self.drop = nn.Dropout(0.4)
        self.w_qinter = nn.Linear(self.D_hidden, self.D_hidden)
        self.W = nn.Linear(self.D_hidden,1)
        self.w_out = nn.Linear(self.D_hidden, self.D_hidden)
        self.relu = nn.ReLU()
    def forward(self, feature, mask_intra, umask):
        '''对inter-speaker做说话人间状态计算'''
        feature = self.drop(self.relu(self.init_trans(feature)))
        v_inter1 = torch.zeros_like(feature)
        for i in range(feature.size(0)):
            c = feature.clone()
            for j in range (feature.size(1)):
                if j == 0:
                    v_inter1[i,j,:] = c[i,j,:]
                    for k in range(j+1,feature.size(1)):
                        if mask_intra[i, j, k] != mask_intra[i, j, j]:
                            v_inter1[i, k, :] = c[i, k, :]
                            break
                    continue
                if umask[i, j] == 0: break
                mark = 0
                for l in range(j-1,-1,-1):#从当前utterance出发找到上一个同说话者
                    if mask_intra[i, j, l] == 1:#找到上一个同说话者
                        mark = 1
                        ki = v_inter1[i, l:j, :].clone() #说话者间的局部信息
                        q_inter = self.w_qinter(c[i,j,:].unsqueeze(0)).permute(1,0)#q_inter:h×1
                        alpha_inter = F.softmax(self.W(torch.mul(q_inter, ki.permute(1,0)).permute(1,0)), dim=0).permute(1,0)#alpha_inter:1×l
                        v_inter1[i, j, :] = torch.tanh(torch.matmul(alpha_inter, ki).squeeze(0))
                        break
                if mark == 0: v_inter1[i, j, :] = c[i, j, :]
        new_feature = self.relu(torch.mul(self.w_out(feature),v_inter1)) + feature
        return new_feature

class LIE(nn.Module):
    def __init__(self, D_hidden, windowp, windowf):
        super().__init__()
        self.D_hidden = D_hidden
        self.windowp = windowp
        self.windowf = windowf
        self.wq = nn.Linear(self.D_hidden, self.D_hidden)
        self.w2 = nn.Linear(self.D_hidden, 1)
        self.wm = nn.Linear(self.D_hidden, self.D_hidden)
        self.norm = nn.LayerNorm(self.D_hidden)
        self.W1 = nn.Linear(self.D_hidden, self.D_hidden)
        self.W2 = nn.Linear(self.D_hidden, self.D_hidden)
        self.drop = nn.Dropout(0.4)
    def forward(self, feature):
        q_local = self.wq(feature)
        v_local = torch.zeros_like(feature)
        for i in range(feature.size(1)):
            if i<self.windowp:
                v_local[:, i, :] = feature[:, i, :]
                continue
            # if i>feature.size(1)-self.windowf:
            #     ki = v_local[:, i - self.windowp:i+1, :].clone()  # 局部utterance信息
            #     alpha_local = F.softmax(self.w2(torch.mul(q_local[:, i, :].unsqueeze(1), ki)),
            #                             dim=1)  # (32,l,1)#得到局部utterance的权重
            #     v_local[:, i, :] = torch.matmul(ki.transpose(1, 2), alpha_local).squeeze(2)  # 乘上权重得到更新后的局部信息
            #     continue
            ki = v_local[:, i-self.windowp:i+1, :].clone()#局部utterance信息
            alpha_local = F.softmax(self.w2(torch.mul(q_local[:, i ,:].unsqueeze(1), ki)), dim=1)#(32,l,1)#得到局部utterance的权重
            v_local[:, i, :] = torch.matmul(ki.transpose(1,2), alpha_local).squeeze(2)#乘上权重得到更新后的局部信息
        out_feature = self.norm(torch.mul(torch.tanh(v_local), feature) + feature)
        # out_feature = self.norm(temp_feature + self.drop(self.W2(F.relu(self.W1(temp_feature)))))
        # out_feature = torch.mul(torch.tanh(v_local), feature) + feature
        return out_feature

class selfmodel(nn.Module):
    def __init__(self,n_speakers,n_classes=6,dropout=0.5,no_cuda=False,D_audio=384,D_text=768,
                 D_hidden=200,dataset='IEMOCAP',use_crn_speaker=True,speaker_weights='1-1-1',D_e=100,att_head=8,args = None):
        super(selfmodel, self).__init__()
        self.args = args
        self.n_speakers = n_speakers
        self.n_classes = n_classes
        self.dropout = dropout
        self.no_cuda = no_cuda
        self.D_text = D_text
        self.att_head = att_head
        self.D_audio = D_audio
        self.D_e = D_e
        self.D_hidden = D_hidden
        self.use_crn_speaker = use_crn_speaker
        self.speaker_weights = list(map(float, speaker_weights.split('-')))
        self.dataset = dataset
        self.conv1d_t = nn.Conv1d(D_text,D_hidden,1)
        self.conv1d_a = nn.Conv1d(D_audio,D_hidden,1)
        # self.spk_embed = nn.Embedding(n_speakers, D_hidden)

        self.rnn_parties_t = nn.GRU(input_size=self.D_hidden, hidden_size=self.D_e, num_layers=2, bidirectional=True,
                                    dropout=self.dropout)
        self.rnn_parties_a = nn.GRU(input_size=self.D_hidden, hidden_size=self.D_e, num_layers=2, bidirectional=True,
                                    dropout=self.dropout)
        self.maskCMT_a2t_1 = CrossModalTransformer2(self.D_hidden,n_heads=4,n_layers=3)
        self.maskCMT_a2t_2 = CrossModalTransformer2(self.D_hidden, n_heads=4, n_layers=3)
        self.maskCMT_a2t_3 = CrossModalTransformer2(self.D_hidden, n_heads=4, n_layers=3)
        self.maskCMT_t2a_1 = CrossModalTransformer2(self.D_hidden, n_heads=4, n_layers=3)
        self.maskCMT_t2a_2 = CrossModalTransformer2(self.D_hidden, n_heads=4, n_layers=3)
        self.maskCMT_t2a_3 = CrossModalTransformer2(self.D_hidden, n_heads=4, n_layers=3)


        self.csgru_a = CS_GRU_v1(input_size=self.D_hidden, hidden_size=self.D_hidden, num_layers=3, bias=None,
                            output_size=self.D_hidden)
        self.csgru_t = CS_GRU_v1(input_size=self.D_hidden, hidden_size=self.D_hidden, num_layers=3, bias=None,
                            output_size=self.D_hidden)
        self.csgru_fu = CS_GRU_v1(input_size=self.D_hidden, hidden_size=self.D_hidden, num_layers=3, bias=None,
                             output_size=self.D_hidden)


        # self.GAT_nheads = 2
        # multi_gats_a = [GAT_dialoggcn_v1(self.D_hidden) for _ in range(self.GAT_nheads)]
        # self.multi_gats_a = nn.ModuleList(multi_gats_a)
        #
        # GATs_a = []
        # for _ in range(args.GAT_nlayers):
        #     # gats += [GAT_dialoggcn(args.hidden_dim)]
        #     GATs_a += [self.multi_gats_a]
        # self.gats_a = nn.ModuleList(GATs_a)
        #
        # GAT_lina = []
        # for _ in range(args.GAT_nlayers):
        #     GAT_lina +=[nn.Linear(self.D_hidden * 2, self.D_hidden)]
        # self.GAT_lina = nn.ModuleList(GAT_lina)
        #
        # GAT_lint = []
        # for _ in range(args.GAT_nlayers):
        #     GAT_lint += [nn.Linear(self.D_hidden * 2, self.D_hidden)]
        # self.GAT_lint = nn.ModuleList(GAT_lint)
        #
        # GAT_linfu = []
        # for _ in range(args.GAT_nlayers):
        #     GAT_linfu += [nn.Linear(self.D_hidden * 2, self.D_hidden)]
        # self.GAT_linfu = nn.ModuleList(GAT_linfu)

        GATs = []
        for _ in range(args.GAT_nlayers):
            # gats += [GAT_dialoggcn(args.hidden_dim)]
            GATs += [GAT_dialoggcn_v1(self.D_hidden)]
        self.gats = nn.ModuleList(GATs)

        GATs_a = []
        for _ in range(args.GAT_nlayers):
            # gats += [GAT_dialoggcn(args.hidden_dim)]
            GATs_a += [GAT_dialoggcn_v1(self.D_hidden)]
        self.gats_a = nn.ModuleList(GATs_a)

        GATs_t = []
        for _ in range(args.GAT_nlayers):
            # gats += [GAT_dialoggcn(args.hidden_dim)]
            GATs_t += [GAT_dialoggcn_v1(self.D_hidden)]
        self.gats_t = nn.ModuleList(GATs_t)

        GATs_fu = []
        for _ in range(args.GAT_nlayers):
            # gats += [GAT_dialoggcn(args.hidden_dim)]
            GATs_fu += [GAT_dialoggcn_v1(self.D_hidden)]
        self.gats_fu = nn.ModuleList(GATs_fu)

        grus_a = []
        for _ in range(args.GAT_nlayers):
            grus_a += [nn.GRUCell(self.D_hidden, self.D_hidden)]
        self.grus_a = nn.ModuleList(grus_a)
        grus_t = []
        for _ in range(args.GAT_nlayers):
            grus_t += [nn.GRUCell(self.D_hidden, self.D_hidden)]
        self.grus_t = nn.ModuleList(grus_t)
        grus_fu = []
        for _ in range(args.GAT_nlayers):
            grus_fu += [nn.GRUCell(self.D_hidden, self.D_hidden)]
        self.grus_fu = nn.ModuleList(grus_fu)

        self.mlp_layer = 2
        layers = [nn.Linear(self.D_hidden*6, self.D_hidden), nn.ReLU()]
        for _ in range(self.mlp_layer - 1):
            layers += [nn.Linear(self.D_hidden, self.D_hidden), nn.ReLU()]
        layers += [nn.Dropout(self.dropout)]
        layers += [nn.Linear(self.D_hidden, n_classes)]
        self.out_mlp = nn.Sequential(*layers)

        self.layer_norma = nn.LayerNorm(self.D_hidden)
        self.layer_normt = nn.LayerNorm(self.D_hidden)
        self.layer_normfu = nn.LayerNorm(self.D_hidden)

        self.wa1 = nn.Linear(self.D_hidden,self.D_hidden)
        self.wa2 = nn.Linear(self.D_hidden,self.D_hidden)
        self.drp = nn.Dropout(self.dropout)

        self.wt1 = nn.Linear(self.D_hidden, self.D_hidden)
        self.wt2 = nn.Linear(self.D_hidden, self.D_hidden)

        self.wfu1 = nn.Linear(self.D_hidden, self.D_hidden)
        self.wfu2 = nn.Linear(self.D_hidden, self.D_hidden)

        self.fuse_lin = nn.Linear(self.D_hidden*6, self.D_hidden)

        self.tempfulin = nn.Linear(self.D_hidden*2, self.D_hidden)
        self.lna = nn.LayerNorm(self.D_hidden)
        self.lnt = nn.LayerNorm(self.D_hidden)
        self.lnfu = nn.LayerNorm(self.D_hidden)

        csgrucell_a = []
        for _ in range(args.GAT_nlayers):
            csgrucell_a += [GRUCell(input_size=self.D_hidden, hidden_size=self.D_hidden)]
        self.csgrucell_a = nn.ModuleList(csgrucell_a)
        # self.lnacell = nn.LayerNorm(self.D_hidden)
        # self.lntcell = nn.LayerNorm(self.D_hidden)
        # self.lnfucell = nn.LayerNorm(self.D_hidden)

        csgrucell_t = []
        for _ in range(args.GAT_nlayers):
            csgrucell_t += [GRUCell(input_size=self.D_hidden, hidden_size=self.D_hidden)]
        self.csgrucell_t = nn.ModuleList(csgrucell_t)

        csgrucell_fu = []
        for _ in range(args.GAT_nlayers):
            csgrucell_fu += [GRUCell(input_size=self.D_hidden, hidden_size=self.D_hidden)]
        self.csgrucell_fu = nn.ModuleList(csgrucell_fu)

        self.out_a = nn.Linear(self.D_hidden, self.D_hidden)
        self.out_t = nn.Linear(self.D_hidden, self.D_hidden)
        self.out_fu = nn.Linear(self.D_hidden, self.D_hidden)
        self.classify_a = nn.Linear(self.D_hidden, self.n_classes)
        self.classify_t = nn.Linear(self.D_hidden, self.n_classes)
        self.classify_fu = nn.Linear(self.D_hidden, self.n_classes)

        self.out_un = nn.Linear(self.D_hidden*2, self.D_hidden)
        self.classify_un = nn.Linear(self.D_hidden, self.n_classes)


        '''测试部分'''
        # self.testLin = nn.Linear(6, 2)


    def forward(self,text,qmask,s_mask,umask,length,audio,mask_intra, mask_inter,mask_local,adj):#text(78,16,768),umask(78,16),adj(32,78,78)
        text = text.permute(1,2,0)
        audio = audio.permute(1,2,0)
        text = self.conv1d_t(text).permute(2,0,1)#(78,32,200)
        audio = self.conv1d_a(audio).permute(2,0,1)#(78,32,200)
        spk_idx = torch.argmax(qmask, dim=-1)
        # spk_vec = self.spk_embed(spk_idx).permute(1,0,2)#speaker embedding

        unprocessed = torch.cat([text.permute(1,0,2), audio.permute(1,0,2)],dim=2)#(66,16,768*2)

        if self.use_crn_speaker:

            # (32,21,200) (32,21,9)
            U_, qmask_ = text.transpose(0, 1), qmask.transpose(0, 1)  # U_(32,77,200),qmask(32,77,2)
            U_p_ = torch.zeros(U_.size()[0], U_.size()[1], 200).type(text.type())  # U_p_(32,77,200)
            U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in
                          range(self.n_speakers)]  # default 2,此时U_parties_是长度为2的list,每个元素都是U_(32,77,200)形状全为0的tensor
            for b in range(U_.size(0)):  # 对每一个batch
                for p in range(len(U_parties_)):  # 对每一个speaker
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(
                        -1)  # index_i即为每个speaker在每个batch中说话的utterance下标，qmask_(32,77,2),torch.nonzero(qmask_[b][:, p])表示取出某一个batch中某一个speaker的第二个维度中非0的元素下标
                    if index_i.size(0) > 0:  # index_i:(29,)即为每个speaker在每个batch中说话的utterance下标
                        U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]  # 有效数范围是(2,32,29)29为index_i中非0的下标元素
                        '''这里把每个batch中每个讲话者说话的utterance下标取出，然后取出对应的utterance特征，放在U_parties_中，
                           U_parties_是长为2的list，每个元素是2个speaker对应的utterance特征'''

            E_parties_ = [self.rnn_parties_t(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in
                          range(len(U_parties_))]
            '''self.rnn_parties(U_parties_[p].transpose(0, 1))输出即GRU输出，(77,32,200)和(4,32,100),
            E_parties_为长度2的list,分别为每个speaker的embedding特征'''

            for b in range(U_p_.size(0)):  # 对每一个batch
                for p in range(len(U_parties_)):  # 对每一个speaker
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)  # 这里是找每个speaker说话的utterance下标
                    if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
            # (21,32,200)
            '''到这里为止相当于利用qmask的索引对输入的特征进行了一次重构，相比于直接加上speaker embedding，这样得到的特征更加精确'''
            U_p = U_p_.transpose(0, 1)
            emotion_t = text + self.speaker_weights[0] * U_p  # emotion_a(78,32,200)
            '''U_p即计算得到的speaker embedding，这里乘上speaker权重后和经过全连接层的语音特征相加'''

            # (32,21,200) (32,21,9)
            U_, qmask_ = audio.transpose(0, 1), qmask.transpose(0, 1)  # U_(32,77,200),qmask(32,77,2)
            U_p_ = torch.zeros(U_.size()[0], U_.size()[1], 200).type(audio.type())  # U_p_(32,77,200)
            U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in
                          range(self.n_speakers)]  # default 2,此时U_parties_是长度为2的list,每个元素都是U_(32,77,200)形状全为0的tensor
            for b in range(U_.size(0)):  # 对每一个batch
                for p in range(len(U_parties_)):  # 对每一个speaker
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(
                        -1)  # index_i即为每个speaker在每个batch中说话的utterance下标，qmask_(32,77,2),torch.nonzero(qmask_[b][:, p])表示取出某一个batch中某一个speaker的第二个维度中非0的元素下标
                    if index_i.size(0) > 0:  # index_i:(29,)即为每个speaker在每个batch中说话的utterance下标
                        U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]  # 有效数范围是(2,32,29)29为index_i中非0的下标元素
                        '''这里把每个batch中每个讲话者说话的utterance下标取出，然后取出对应的utterance特征，放在U_parties_中，
                           U_parties_是长为2的list，每个元素是2个speaker对应的utterance特征'''

            E_parties_ = [self.rnn_parties_a(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in
                          range(len(U_parties_))]
            '''self.rnn_parties(U_parties_[p].transpose(0, 1))输出即GRU输出，(77,32,200)和(4,32,100),
            E_parties_为长度2的list,分别为每个speaker的embedding特征'''

            for b in range(U_p_.size(0)):  # 对每一个batch
                for p in range(len(U_parties_)):  # 对每一个speaker
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)  # 这里是找每个speaker说话的utterance下标
                    if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
            # (21,32,200)
            '''到这里为止相当于利用qmask的索引对输入的特征进行了一次重构，相比于直接加上speaker embedding，这样得到的特征更加精确'''
            U_p = U_p_.transpose(0, 1)
            emotion_a = audio + self.speaker_weights[0] * U_p    #emotion_a(78,32,200)
            '''U_p即计算得到的speaker embedding，这里乘上speaker权重后和经过全连接层的语音特征相加'''


        emotion_a = audio.permute(1, 0, 2)
        emotion_t = text.permute(1, 0, 2)#(32,78,200)

        '''实验：只使用三个CMT，只使用t2a'''
        cross_a2t_1, _ = self.maskCMT_a2t_1(emotion_a, emotion_t, None)#第一个说话者的文本到语音跨模态交互,(78,32,200)
        cross_a2t_2, _ = self.maskCMT_a2t_2(emotion_a, emotion_t, None)
        cross_a2t_3, _ = self.maskCMT_a2t_3(emotion_a, emotion_t, None)

        cross_a2t_3 = self.csgru_a(cross_a2t_3, cross_a2t_2, cross_a2t_1)
        # lie_a = LIE(D_hidden=self.D_hidden,windowp=4, windowf=4).to(device)
        # cross_a2t_3 = lie_a(cross_a2t_3)
        cross_a2t = torch.cat([cross_a2t_1, cross_a2t_2, cross_a2t_3], dim=2)  # (32,78,600)

        cross_t2a_1, _ = self.maskCMT_t2a_1(emotion_t, emotion_a, None)
        cross_t2a_2, _ = self.maskCMT_t2a_2(emotion_t, emotion_a, None)
        cross_t2a_3, _ = self.maskCMT_t2a_3(emotion_t, emotion_a, None)

        cross_t2a_3 = self.csgru_t(cross_t2a_3, cross_t2a_2, cross_t2a_1)
        # lie_t = LIE(D_hidden=self.D_hidden,windowp=4, windowf=4).to(device)
        # cross_t2a_3 = lie_t(cross_t2a_3)
        cross_t2a = torch.cat([cross_t2a_1, cross_t2a_2, cross_t2a_3],dim=2)  #(32,78,600)

        cross_at = torch.cat([cross_a2t, cross_t2a],dim=2)#(32,78,1200)
        fuse_out = F.relu(self.fuse_lin(cross_at))
        fuse_out = self.drp(fuse_out)
        lie_fu = LIE(D_hidden=self.D_hidden, windowp=7, windowf=7).to(device)
        fuse_out = lie_fu(fuse_out)


        num_utter = fuse_out.size(1)
        H_a0 = cross_a2t_3  #语音模态用的是a2t的local掩码
        H_t0 = cross_t2a_3  #文本模态用的是t2a的local掩码
        H_fu0 = fuse_out
        H_a = [H_a0]#H_a0即初始的所有节点特征
        H_t = [H_t0]
        H_fu = [H_fu0]
        adj_a = adj.clone()
        adj_t = adj.clone()
        adj_fu = adj.clone()
        for l in range(self.args.GAT_nlayers - 1):
            H1_a = self.grus_a[l](H_a[l][:, 0, :]).unsqueeze(1)  # (32,1,200)初始化和第一个utterance形状相同的特征
            H1_t = self.grus_t[l](H_t[l][:, 0, :]).unsqueeze(1)
            H1_fu = self.grus_fu[l](H_fu[l][:, 0, :]).unsqueeze(1)
            z0_a = H1_a
            z0_t = H1_t
            z0_fu = H1_fu
            '''如果要做过去p个和未来f个，可考虑将H1_a为列表，每个存储不同层上下文信息，当前使用上一层信息，'''
            for i in range(1, num_utter):
                # test = self.out_a(H_a[l][:, i, :])*self.out_t(H_a[l][:, 50, :])
                # test = torch.sum(test,dim=1)
                # test = F.gumbel_softmax(test,0.5,True,1)
                # test = (test>0.1).float()
                att_weight_a, att_sum_a_temp = self.gats_a[l](H_a[l][:, i, :], H1_t, H1_t, adj_a[:, i, :i], s_mask[:, i, :i])  # 用text对应节点来更新audio特征
                att_weight_t, att_sum_t_temp = self.gats_t[l](H_t[l][:, i, :], H1_a, H1_a, adj_t[:, i, :i], s_mask[:, i, :i]) # 反之用audio节点来更新text对应节点特征
                att_weight_fu, att_sum_fu_temp = self.gats_fu[l](H_fu[l][:, i, :], H1_fu, H1_fu, adj_fu[:, i, :i], s_mask[:, i, :i])
                att_sum_a = self.csgrucell_a[l](att_sum_a_temp, att_sum_t_temp, att_sum_fu_temp, H_a[l][:, i, :])
                att_sum_t = self.csgrucell_t[l](att_sum_t_temp, att_sum_a_temp, att_sum_fu_temp, H_t[l][:, i, :])
                att_sum_fu = self.csgrucell_fu[l](att_sum_fu_temp, att_sum_t_temp, att_sum_a_temp, H_fu[l][:, i, :])
                H1_a = torch.cat([H1_a, att_sum_a.unsqueeze(1)], dim=1)  # 将更新后的每个utterance特征在第一个维度拼接
                H1_t = torch.cat([H1_t, att_sum_t.unsqueeze(1)], dim=1)
                H1_fu = torch.cat([H1_fu, att_sum_fu.unsqueeze(1)], dim=1)

            H1_a = self.layer_norma(H_a[l] + H1_a)  # 将GAT的输出进行残差连接,层归一化
            H1_a = self.layer_norma(H1_a + self.drp(self.wa2(F.relu(self.wa1(H1_a)))))
            H1_t = self.layer_normt(H_t[l] + H1_t)
            H1_t = self.layer_normt(H1_t + self.drp(self.wt2(F.relu(self.wt1(H1_t)))))
            H1_fu = self.layer_normfu(H_fu[l] + H1_fu)
            H1_fu = self.layer_normfu(H1_fu + self.drp(self.wfu2(F.relu(self.wfu1(H1_fu)))))
            H_a.append(H1_a)
            H_t.append(H1_t)
            H_fu.append(H1_fu)

        '''想法:分别用文本和语音预测输出来做KL散度损失，最后拼接做预测'''
        feature_a = torch.cat(H_a, dim=2)#(32,78,600)
        feature_t = torch.cat(H_t, dim=2)#(32,78,600)
        feature_fu = torch.cat(H_fu, dim=2)#(32,78,600)
        feature = torch.cat([feature_t, feature_a, feature_fu], dim=2)#(32,78,1800)

        #用text和audio原始特征的直接拼接来预测情绪做t-sne可视化
        raw_cat_feature = self.out_un(torch.cat([text, audio],dim=2))
        raw_cat_feature = raw_cat_feature.permute(1, 0, 2)
        pred_raw = []
        for l in range(raw_cat_feature.size(0)):
            temp = raw_cat_feature[l, :length[l], :]
            pred_raw.append(temp)
        temp_raw = []
        for i in range(len(pred_raw)):
            if i == 0: temp_raw = pred_raw[i]
            if i == len(pred_raw) - 1: break
            temp_raw = torch.cat([temp_raw, pred_raw[i + 1]], dim=0)
        prob_raw = self.classify_fu(temp_raw)
        log_prob_raw = F.log_softmax(prob_raw,dim=1)

        pred_a = []
        for l in range(feature.size(0)):
            temp = cross_a2t_3[l, :length[l], :]
            pred_a.append(temp)
        temp_a = []
        for i in range(len(pred_a)):
            if i==0: temp_a = pred_a[i]
            if i==len(pred_a) - 1: break
            temp_a = torch.cat([temp_a, pred_a[i+1]], dim=0)
        prob_a = self.classify_a(F.relu(self.out_a(temp_a)))
        log_prob_a = F.log_softmax(prob_a, dim=1)
        unlog_prob_a = F.softmax(prob_a,dim=1)

        pred_t = []
        for l in range(feature.size(0)):
            temp = cross_t2a_3[l, :length[l], :]
            pred_t.append(temp)
        temp_t = []
        for i in range(len(pred_t)):
            if i == 0: temp_t = pred_t[i]
            if i == len(pred_t) - 1: break
            temp_t = torch.cat([temp_t, pred_t[i + 1]], dim=0)
        prob_t = self.classify_t(F.relu(self.out_t(temp_t)))
        log_prob_t = F.log_softmax(prob_t, dim=1)
        unlog_prob_t = F.softmax(prob_t, dim=1)

        pred_fu = []
        for l in range(feature.size(0)):
            temp = fuse_out[l, :length[l], :]
            pred_fu.append(temp)
        temp_fu = []
        for i in range(len(pred_fu)):
            if i == 0: temp_fu = pred_fu[i]
            if i == len(pred_fu) - 1: break
            temp_fu = torch.cat([temp_fu, pred_fu[i + 1]], dim=0)
        prob_fu = self.classify_fu(F.relu(self.out_fu(temp_fu)))
        log_prob_fu = F.log_softmax(prob_fu, dim=1)
        unlog_prob_fu = F.softmax(prob_fu, dim=1)

        pred = []
        for l in range(feature.size(0)):
            temp = feature[l, :length[l], :]
            pred.append(temp)
        temp2 = []
        for i in range(len(pred)):
            if i==0: temp2 = pred[i]
            if i==len(pred) - 1: break
            temp2 = torch.cat([temp2, pred[i+1]], dim=0)
        log_prob = self.out_mlp(temp2)
        unlog_prob = F.softmax(log_prob, dim=1)
        log_prob = F.log_softmax(log_prob, dim=1)

        pred_un = []
        for l in range(unprocessed.size(0)):
            temp = unprocessed[l, :length[l], :]
            pred_un.append(temp)
        temp_un= []
        for i in range(len(pred_un)):
            if i == 0: temp_un = pred_un[i]
            if i == len(pred_un) - 1: break
            temp_un = torch.cat([temp_un, pred_un[i + 1]], dim=0)
        prob_un = self.classify_un(F.relu(self.out_un(temp_un)))
        log_prob_un = F.log_softmax(prob_un, dim=1)
        unlog_prob_un = F.softmax(prob_un, dim=1)


        # '''测试部分success'''
        # pre_shift_matrix = torch.zeros((log_prob.shape[0], log_prob.shape[0], log_prob.shape[1]), dtype=torch.float32).cuda()
        # for i in range(log_prob.shape[0]):
        #     pre_shift_matrix[i] = log_prob[i:i+1] - log_prob
        #
        # pre_shift_matrix = self.testLin(pre_shift_matrix)

        return log_prob, unlog_prob, log_prob_a, log_prob_t, unlog_prob_a, unlog_prob_t, log_prob_fu, unlog_prob_fu,temp_a,temp_t, temp_fu,log_prob_un,log_prob_raw

        '''# 因为要用y指导x,所以求x的对数概率，y的概率
            logp_x = F.log_softmax(x, dim=-1)
            p_y = F.softmax(y, dim=-1)'''



