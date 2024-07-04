import csv
import json
import os

import numpy as np
import pandas as pd
import pickle

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class IEMOCAPDataset(Dataset):

    def __init__(self, path=None, train=True, args = None):
        self.text_path = 'F:\ERC_CODE\Iemocap_feature/text_feature_paraphrase-distilroberta'
        self.audio_path = 'F:\ERC_CODE\Iemocap_feature\wav2vec2-base-finetuned-iemocap6'
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')
        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)
        self.split = train
        self.args = args
        self.selected_items = {}

    def __getitem__(self, index):
        vid = self.keys[index]

        self.selected_items[vid] = self.videoLabels[vid]
        sentence = self.videoSentence[vid]
        label = self.videoLabels[vid]

        audio = []
        for file in self.videoIDs[vid]:  # 读取当前对话的所有文件
            with open(os.path.join(self.audio_path, vid, file + '.csv'), 'r') as csvfile:
                reader = csv.reader(csvfile)
                audio.append(list(reader)[0])
        audio = np.array(audio, dtype=float)
        # audio = (audio - np.mean(audio)) / np.std(audio)

        text = []
        for file in self.videoIDs[vid]:  # 读取当前对话的所有文件
            with open(os.path.join(self.text_path, vid, file + '.csv'), 'r') as csvfile:
                reader = csv.reader(csvfile)
                text.append(list(reader)[0])
        text = np.array(text, dtype=float)
        # text = (text - np.mean(text)) / np.std(text)

        for i,x in enumerate(self.videoSpeakers[vid]):
            if i==0:
                speaker = []
            if x=='M':
                speaker.append(0)
            else:
                speaker.append(1)
        spk_list = speaker

        return torch.FloatTensor(text),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(audio),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               self.videoSpeakers[vid],\
               spk_list,vid

    def __len__(self):
        return self.len

    def get_s_mask(self, speakers, max_dialog_len):
        '''
        :param speakers:
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        s_mask = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype = torch.long)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i,j] = 1
            s_mask.append(s)
        return torch.stack(s_mask)

    def get_mask_intra(self,speakers,max_dialog_len):#mask_intra:相同说话者对应全为1,获得同一说话人掩码(32,78,78),(78,78)为对角矩阵
        mask_intra = []
        for speaker in speakers:
            mask1 = torch.zeros(max_dialog_len, max_dialog_len)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i]==speaker[j]:
                        mask1[i,j] = 1
            mask_intra.append(mask1)
        return torch.stack(mask_intra)

    def get_mask_intra_v2(self, speakers, max_dialog_len):
        mask_intra = []
        for speaker in speakers:
            mask1 = torch.zeros(max_dialog_len, max_dialog_len)
            for i,s in enumerate(speaker):
                countp = 0
                countf = 0
                mask1[i, i] = 1
                for j in range(i - 1, -1, -1):
                    if speaker[j] == s:  # 如果utterance j和当前utterance是同一人
                        mask1[i, j] = 1
                        countp += 1
                        if countp == 2:
                            break
                for k in range(i + 1, len(speaker)):
                    if speaker[k] == s:  # 如果utterance j和当前utterance是同一人
                        mask1[i, k] = 1
                        countf += 1
                        if countf == 2:
                            break
            mask_intra.append(mask1)
        return torch.stack(mask_intra)

    def get_mask_inter(self,speakers,max_dialog_len):#mask_inter:不同说话者对应全为1
        mask_inter = []
        for speaker in speakers:
            mask2 = torch.zeros(max_dialog_len, max_dialog_len)
            for i in range(len(speaker)):
                mask2[i, i] = 1
                for j in range(len(speaker)):
                    if speaker[i] != speaker[j]:
                        mask2[i, j] = 1
            mask_inter.append(mask2)
        return torch.stack(mask_inter)

    def get_mask_inter_v2(self,speakers,max_dialog_len):#mask_inter:不同说话者对应全为1
        '''inter-speaker:找到上一个相同speaker后将上一个同speaker到当前speaker之间的所有utterance置为1，包括上一个speaker所在utterance
        但不包括当前utterance'''
        mask_inter = []
        for speaker in speakers:
            mask2 = torch.zeros(max_dialog_len, max_dialog_len)
            for i in range(len(speaker)):
                for k in range(i - 1, -1, -1):  # 找到上一个相同说话者之间的所有不同utterance
                    if speaker[i] != speaker[k]:
                        mask2[i, k] = 1
                    else:
                        mask2[i, k] = 1
                        break
                for j in range(i + 1, len(speaker)):  # 找到上一个相同说话者之间的所有不同utterance
                    if speaker[i] != speaker[j]:
                        mask2[i, j] = 1
                    else:
                        mask2[i, j] = 1
                        break
            mask_inter.append(mask2)
        return torch.stack(mask_inter)

    def get_mask_localinter(self,speakers,max_dialog_len):
        '''只关注与当前说话者相同的上一个说话者与下一个相同说话者之间的utterance，并且包含上一个说话者和下一个说话者和自己'''
        mask_inter = []
        for speaker in speakers:
            mask2 = torch.zeros(max_dialog_len, max_dialog_len)
            for i in range(len(speaker)):
                mask2[i, i] = 1
                if i!=0:
                    for k in range(i-1, -1, -1):#找到上一个相同说话者之间的所有不同utterance
                        if speaker[i] != speaker[k]:
                            mask2[i, k] = 1
                        else:
                            mask2[i, k] = 1
                            break
                if i!= len(speaker)-1:
                    for j in range(i+1,len(speaker)):##找到下一个相同说话者之间的所有不同utterance
                        if speaker[i] != speaker[j]:
                            mask2[i, j] = 1
                        else:
                            mask2[i, j] = 1
                            break
            mask_inter.append(mask2)
        return torch.stack(mask_inter)

    def get_mask_local(self,speakers,max_dialog_len):#局部信息mask，只关注过去p个utterance和未来f个utterance
        mask_local = []
        for speaker in speakers:
            mask3 = torch.zeros(max_dialog_len, max_dialog_len)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if abs(i-j) <= 21:#这里暂时只设置关注过去和未来2个utterance
                        mask3[i,j] = 1
            mask_local.append(mask3)
        return torch.stack(mask_local)

    def get_mask_local_v2(self,speakers,max_dialog_len):#局部信息mask，只关注过去p个utterance和未来f个utterance
        mask_local = []
        for speaker in speakers:
            mask3 = torch.zeros(max_dialog_len, max_dialog_len)
            for i,s in enumerate(speaker):
                countp = 0
                countf = 0
                mask3[i, i] = 1
                for j in range(i-1, -1, -1):
                    mask3[i, j] = 1
                    if speaker[j] == s:#如果utterance j和当前utterance是同一人
                        countp += 1
                        if countp==2:
                            break
                for k in range(i+1, len(speaker)):
                    mask3[i, k] = 1
                    if speaker[k] == s:#如果utterance j和当前utterance是同一人
                        countf += 1
                        if countf==1:
                            break

            mask_local.append(mask3)
        return torch.stack(mask_local)


    def get_adj(self, speakers, max_dialog_len):
        '''
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        adj = []
        for speaker in speakers:
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i,s in enumerate(speaker):
                cnt_p = 0
                for j in range(i - 1, -1, -1):   #倒序遍历
                    a[i,j] = 1        #a[1,0]=1,a[2,1]=1,a[2,0]=1,a[3,2]=1
                    if speaker[j] == s:#如果utterance j和当前utterance是同一人,这里的意思是边连接到与当前utterance i的说话者的前windowp个utterance
                        cnt_p += 1
                        if cnt_p==self.args.windowp:
                            break
            adj.append(a)
        return torch.stack(adj)

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        max_dialog_len = max([d[5].size(0) for d in data])
        speakers = [d[6] for d in data]
        mask_intra = self.get_mask_intra([d[6] for d in data],max_dialog_len)
        mask_inter = self.get_mask_inter([d[6] for d in data],max_dialog_len)
        mask_local = self.get_mask_local([d[6] for d in data],max_dialog_len)
        text = pad_sequence([d[0] for d in data])
        visual = pad_sequence([d[1] for d in data])
        audio = pad_sequence([d[2] for d in data])
        spk_onehot = pad_sequence([d[3] for d in data])
        umask = pad_sequence([d[4] for d in data],True)
        label = pad_sequence([d[5] for d in data],True)
        s_mask = self.get_s_mask(speakers,max_dialog_len)
        adj = self.get_adj(speakers,max_dialog_len)
        vid = [d[-1] for d in data]
        # select_items = [d[6] for d in data]
        # speakers = [d[6] for d in dat]
        return text, visual, audio, spk_onehot,s_mask, umask, label,  mask_intra, mask_inter,mask_local, adj, vid
        # return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]


class MELDDataset(Dataset):

    def __init__(self, path=None, train=True,args=None):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid,self.aaa = pickle.load(open(path, 'rb'),encoding='latin1')
        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        # self.audio_data = pickle.load(open('F:\ERC_CODE\meld_feature/meld_dia_audio.pkl', 'rb'), encoding='latin1')
        self.data = self.load_data()

        self.len = len(self.keys)
        self.args = args

    def load_data(self):
        with open('E:\modelcode\Myself_MMDFN\data\meld\wpl_MELD_train_all', encoding='utf-8') as f:
            raw_data = json.load(f)
        return raw_data

    def __getitem__(self, index):
        vid = self.keys[index]
        # print('vid:{}'.format(vid))
        # print('label:{}'.format(len(self.videoLabels[vid])))
        # if vid<60:
        #     print('self.data:{}'.format(len(self.data[vid])))
        # else:
        #     print('self.data:{}'.format(len(self.data[vid -1])))
        if vid<60:
            text_feature_list = np.array([d['cls'] for d in self.data[vid]])  # 拿到对应对话中所有utterance的文本特征
            audio_feature_list = np.array([d['wav'] for d in self.data[vid]])  # 拿到对应对话中所有utterance的文本特征
        else:
            text_feature_list = np.array([d['cls'] for d in self.data[vid-1]])  # 拿到对应对话中所有utterance的文本特征
            audio_feature_list = np.array([d['wav'] for d in self.data[vid-1]])  # 拿到对应对话中所有utterance的文本特征
        # audio_feature_list = np.array(self.audio_data[vid],dtype=float)
        return torch.FloatTensor(text_feature_list),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(audio_feature_list),\
               torch.FloatTensor(self.videoSpeakers[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label+=self.videoLabels[key]
        return return_label

    def get_mask_intra(self,speakers,max_dialog_len):#mask_intra:相同说话者对应全为1,获得同一说话人掩码(32,78,78),(78,78)为对角矩阵
        mask_intra = []
        for speaker in speakers:
            mask1 = torch.zeros(max_dialog_len, max_dialog_len)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i].equal(speaker[j]):
                        mask1[i,j] = 1
            mask_intra.append(mask1)
        return torch.stack(mask_intra)

    def get_mask_inter(self,speakers,max_dialog_len):#mask_inter:不同说话者对应全为1
        mask_inter = []
        for speaker in speakers:
            mask2 = torch.zeros(max_dialog_len, max_dialog_len)
            for i in range(len(speaker)):
                mask2[i, i] = 1
                for j in range(len(speaker)):
                    if not speaker[i].equal(speaker[j]):
                        mask2[i, j] = 1
            mask_inter.append(mask2)
        return torch.stack(mask_inter)

    def get_mask_local(self,speakers,max_dialog_len):#局部信息mask，只关注过去p个utterance和未来f个utterance
        mask_local = []
        for speaker in speakers:
            mask3 = torch.zeros(max_dialog_len, max_dialog_len)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if abs(i-j) <= 3:#这里暂时只设置关注过去和未来2个utterance
                        mask3[i,j] = 1
            mask_local.append(mask3)
        return torch.stack(mask_local)

    def get_s_mask(self, speakers, max_dialog_len):
        '''
        :param speakers:
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
        s_mask = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype = torch.long)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i].equal(speaker[j]):
                        s[i,j] = 1
            s_mask.append(s)
        return torch.stack(s_mask)

    def get_adj(self, speakers, max_dialog_len):
        '''
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        adj = []
        for speaker in speakers:
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i,s in enumerate(speaker):
                cnt_p = 0
                for j in range(i - 1, -1, -1):   #倒序遍历
                    a[i,j] = 1        #a[1,0]=1,a[2,1]=1,a[2,0]=1,a[3,2]=1
                    if speaker[j].equal(speaker[i]):#如果utterance j和当前utterance是同一人,这里的意思是边连接到与当前utterance i的说话者的前windowp个utterance
                        cnt_p += 1
                        if cnt_p==self.args.windowp:
                            break
            adj.append(a)
        return torch.stack(adj)

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        speakers = [d[3] for d in data]
        text = pad_sequence([d[0] for d in data])
        visual = pad_sequence([d[1] for d in data])
        audio = pad_sequence([d[2] for d in data])
        spk_onehot = pad_sequence([d[3] for d in data])
        max_dialog_len = max([d[5].size(0) for d in data])
        mask_intra = self.get_mask_intra(speakers,max_dialog_len)
        mask_inter = self.get_mask_inter(speakers,max_dialog_len)
        mask_local = self.get_mask_local(speakers,max_dialog_len)
        s_mask = self.get_s_mask(speakers,max_dialog_len)
        umask = pad_sequence([d[4] for d in data], True)
        adj = self.get_adj(speakers, max_dialog_len)
        label = pad_sequence([d[5] for d in data], True)
        vid = [d[-1] for d in data]
        # return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]
        return text, visual, audio, spk_onehot, s_mask, umask, label, mask_intra, mask_inter, mask_local, adj, vid


class DailyDialogueDataset(Dataset):

    def __init__(self, split, path):
        
        self.Speakers, self.Features, \
        self.ActLabels, self.EmotionLabels, self.trainId, self.testId, self.validId = pickle.load(open(path, 'rb'))
        
        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]

        self.len = len(self.keys)

    def __getitem__(self, index):
        conv = self.keys[index]
        
        return  torch.FloatTensor(self.Features[conv]), \
                torch.FloatTensor([[1,0] if x=='0' else [0,1] for x in self.Speakers[conv]]),\
                torch.FloatTensor([1]*len(self.EmotionLabels[conv])), \
                torch.LongTensor(self.EmotionLabels[conv]), \
                conv

    def __len__(self):
        return self.len
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<2 else pad_sequence(dat[i], True) if i<4 else dat[i].tolist() for i in dat]
