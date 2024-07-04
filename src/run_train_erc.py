import argparse
import numpy as np
import datetime
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import ticker
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from dataloader import IEMOCAPDataset, MELDDataset
from model import LSTMModel, GRUModel, DialogRNNModel, DialogueGNNModel
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from src.myselfmodel import selfmodel

from loss import FocalLoss, MaskedNLLLoss

seed = 2021


def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def convert_feature2tsne(feature,length):
    feature_list = []
    for l in range(feature.size(1)):
        temp = feature[ :length[l], l, :]
        feature_list.append(temp)
    temp1 = []
    for i in range(len(feature_list)):
        if i == 0: temp1 = feature_list[i]
        if i == len(feature_list) - 1: break
        temp1 = torch.cat([temp1, feature_list[i + 1]], dim=0)
    return temp1

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_MELD_loaders(data_path=None, batch_size=32, valid_rate=0.1, num_workers=0, pin_memory=False,args=None):
    trainset = MELDDataset(data_path,args=args)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid_rate)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset(data_path, train=False,args=args)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(data_path=None, batch_size=32, valid_rate=0.1, num_workers=0, pin_memory=False,args=None):
    trainset = IEMOCAPDataset(path=data_path, train=True, args=args)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid_rate)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(path=data_path, train=False,args=args)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

def loss_task(input, label):#log_prob(1491,6),label(1491,)
    target = label.view(-1, 1)
    logpt = torch.log_softmax(input, dim=-1)
    logpt = logpt.gather(1, target)  # (1491,1)
    loss = -1 * logpt.mean()
    return loss

# def loss_KLDiv(pred,target):
#     log_pred = torch.log_softmax(pred, dim=1)
#     log_target = torch.log_softmax(target, dim=1)
#     losses = torch.sum(torch.exp(log_target) * (log_target - log_pred), dim=1)
#     return losses.mean()


def train_or_eval_graph_model(model, loss_f, dataloader, epoch=0, train_flag=False, optimizer=None, cuda_flag=False, target_names=None,
                              test_label=False, tensorboard=False):
    losses, preds, labels, raw_preds = [], [], [], []
    scores, vids = [], []

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []

    if cuda_flag: ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    assert not train_flag or optimizer != None
    if train_flag:
        model.train()
    else:
        model.eval()

    seed_everything()
    listfeature_a = []
    listfeature_t = []
    listfeature_fu = []
    raw_text = []
    raw_audio = []
    raw_fu = []
    for data in dataloader:
        if train_flag:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask,s_mask, umask, label, mask_intra, mask_inter, mask_local, adj = [d.cuda() for d in data[:11]] if cuda_flag else data[:-1]
        vid = data[11]
        # select_items = data[11]
        # print(label)
        #获取

        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]  #nonzero()得到非零元素索引下标

        log_prob, unlog_prob, log_prob_a, log_prob_t, unlog_prob_a, unlog_prob_t, log_prob_fu, unlog_prob_fu,feature_a,feature_t, feature_fu,log_prob_un,log_prob_raw = model(textf, qmask, s_mask, umask, lengths, acouf,mask_intra, mask_inter, mask_local, adj) #textf(77,32,100),qmask(77,32,2),umask(32,77),acouf(77,32,1582),visuf(77,32,342)

        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        #生成情绪转移矩阵
        # emo_shift_matrix = torch.zeros((label.shape[0], label.shape[0]),dtype=torch.int64).cuda()
        #
        # for i in range(len(label)):
        #     for j in range(i+1, len(label)):
        #         if(label[i]!=label[j]):
        #             emo_shift_matrix[i][j] = 1
        #             emo_shift_matrix[j][i] = 1

        listfeature_a.append(feature_a)
        listfeature_t.append(feature_t)
        listfeature_fu.append(feature_fu)
        # tsne原始特征
        raw_text.append(convert_feature2tsne(textf,lengths))
        raw_audio.append(convert_feature2tsne(acouf,lengths))
        KL_loss = nn.KLDivLoss(reduction = 'batchmean')
        # loss_test = loss_f(pre_shift_matrix, emo_shift_matrix)
        loss_a = loss_f(log_prob_a, label)
        loss_t = loss_f(log_prob_t, label)
        loss_fu = loss_f(log_prob_fu, label)
        loss_task = loss_f(log_prob, label)
        loss_kl = KL_loss(log_prob_a, unlog_prob) + KL_loss(log_prob_t, unlog_prob) + KL_loss(log_prob_a, unlog_prob_t)\
                  + KL_loss(log_prob_fu, unlog_prob) + KL_loss(log_prob_t,unlog_prob_a)
        loss = loss_task + loss_kl + loss_a + loss_t + loss_fu

        pred_un = []
        raw_preds.append(torch.argmax(log_prob_raw, 1).cpu().numpy())
        pred_un.append(torch.argmax(log_prob_un, 1).cpu().numpy())

        matchs = []
        for index, (elementA, elementB) in enumerate(zip(label, pred_un[0])):
            if elementA == elementB:
                matchs.append(index)

        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())

        if train_flag:
            loss.backward()
            if tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
            # scheduler_1.step()
    list_a = torch.cat(listfeature_a,dim=0).cpu().detach().numpy()
    list_t = torch.cat(listfeature_t, dim=0).cpu().detach().numpy()
    list_fu = torch.cat(listfeature_fu, dim=0).cpu().detach().numpy()
    raw_list_audio = torch.cat(raw_audio,dim=0).cpu().detach().numpy()
    raw_list_text = torch.cat(raw_text,dim=0).cpu().detach().numpy()
    if preds != []:
        preds = np.concatenate(preds)
        preds_un = np.concatenate(pred_un)
        raw_preds = np.concatenate(raw_preds)
        labels = np.concatenate(labels)
    else:
        return [], [], float('nan'), float('nan'), [], [], float('nan'), []

    vids += data[-1]
    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    labels = np.array(labels)
    preds = np.array(preds)
    vids = np.array(vids)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    all_each = metrics.classification_report(labels, preds, target_names=target_names, digits=4)#用来展示表格形式数据
    # all_each = None
    all_acc = ["ACC"]
    for i in range(len(target_names)):
        all_acc.append("{}: {:.4f}".format(target_names[i], accuracy_score(labels[labels == i], preds[labels == i])))


    return all_each, all_acc, avg_loss, avg_accuracy, labels, preds, avg_fscore, [vids, ei, et, en, el],list_a,list_t,list_fu,raw_list_audio,raw_list_text,preds,raw_preds


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--dataset', default='IEMOCAP', help='dataset to train and test')
    parser.add_argument('--data_dir', type=str, default='../data/iemocap/IEMOCAP_features.pkl', help='dataset dir')
    parser.add_argument('--graph_model', action='store_true', default=True, help='whether to use graph model after recurrent encoding')
    parser.add_argument('--use_residue', action='store_true', default=True, help='whether to use residue information or not')
    parser.add_argument('--use_crn_speaker', action='store_true', default=True, help='whether to use use crn_speaker embedding') #是否做speaker_embedding
    parser.add_argument('--speaker_weights', type=str, default='1-1-1', help='speaker weight 0-0-0')
    parser.add_argument('--reason_flag', action='store_true', default=False, help='reason flag')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--valid_rate', type=float, default=0.1, metavar='valid_rate', help='valid rate, 0.0/0.1')
    parser.add_argument('--GAT_nlayers', type=int, default=2, help='Deep_GCN_nlayers')
    parser.add_argument('--lr', type=float, default=0.0002, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='dropout', help='dropout rate')
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha 0.1/0.2')
    parser.add_argument('--lamda', type=float, default=0.5, help='eta 0.5/0')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma 0.5/1/2')
    parser.add_argument('--windowp', type=int, default=1, help='context window size for constructing edges in graph model for past utterances')
    parser.add_argument('--windowf', type=int, default=1, help='context window size for constructing edges in graph model for future utterances')
    parser.add_argument('--multiheads', type=int, default=6, help='multiheads')
    parser.add_argument('--loss', default="FocalLoss", help='loss function: FocalLoss/NLLLoss')
    parser.add_argument('--class_weight', action='store_true', default=False, help='use class weights')
    parser.add_argument('--save_model_dir', type=str, default='../outputs/iemocap_demo/', help='saved model dir')
    parser.add_argument('--tensorboard', action='store_true', default=True, help='Enables tensorboard log')
    parser.add_argument('--test_label', action='store_true', default=False, help='whether do test only')
    parser.add_argument('--load_model', type=str, default='../outputs/iemocap_demo/model_4.pkl', help='trained model dir')
    parser.add_argument('--nodal_att_type', type=str, default='past', choices=['global', 'past'],
                        help='type of nodal attention')

    args = parser.parse_args()
    today = datetime.datetime.now()
    print(args)
    # if args.av_using_lstm:
    #     name_ = args.modals + '_' + args.graph_type + '_' + args.graph_construct + 'using_lstm_' + args.dataset
    # else:
    #     name_ = args.modals + '_' + args.graph_type + '_' + args.graph_construct + str(args.GAT_nlayers) + '_' + args.dataset
    #
    # if args.use_speaker:
    #     name_ = name_ + '_speaker'
    # if args.use_modal:
    #     name_ = name_ + '_modal'

    cuda_flag = torch.cuda.is_available() and not args.no_cuda

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

    n_epochs = args.epochs
    batch_size = args.batch_size
    # modals = args.modals
    feat2dim = {'IS10': 1582, '3DCNN': 512, 'textCNN': 100, 'bert': 768, 'denseface': 342, 'MELD_text': 600, 'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if args.dataset == 'IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = feat2dim['textCNN'] if args.dataset == 'IEMOCAP' else feat2dim['MELD_text']

    D_m = D_text

    n_speakers, n_classes, class_weights, target_names = -1, -1, None, None
    if args.dataset == 'IEMOCAP':
        n_speakers, n_classes = 2, 6
        target_names = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']
        class_weights = torch.FloatTensor([1 / 0.086747,
                                           1 / 0.144406,
                                           1 / 0.227883,
                                           1 / 0.160585,
                                           1 / 0.127711,
                                           1 / 0.252668]) #平衡权重,用来平衡输入样本不均衡的问题
    if args.dataset == 'MELD':
        n_speakers, n_classes = 9, 7

        target_names = ['neu', 'sur', 'fea', 'sad', 'joy', 'dis', 'ang']
        class_weights = torch.FloatTensor([1.0 / 0.466750766,
                                           1.0 / 0.122094071,
                                           1.0 / 0.027752748,
                                           1.0 / 0.071544422,
                                           1.0 / 0.171742656,
                                           1.0 / 0.026401153,
                                           1.0 / 0.113714183])

    seed_everything()
    if args.dataset == 'IEMOCAP':
        model = selfmodel(n_speakers=2,n_classes=6,dropout=0.5,no_cuda=False,D_audio=768,D_text=768,
                      D_hidden=200,dataset='IEMOCAP',use_crn_speaker=True,speaker_weights='1-1-1',D_e=100,att_head=8,args=args)
    if args.dataset == 'MELD':
        model = selfmodel(n_speakers=9, n_classes=7, dropout=0.5, no_cuda=False, D_audio=1024, D_text=1024,
                          D_hidden=200, dataset='MELD', use_crn_speaker=True, speaker_weights='1-1-1', D_e=100,
                          att_head=8, args=args)

    modals = 'text,audio'
    print("The model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    print('Running on the {} features........'.format(modals))

    if cuda_flag:
        # torch.cuda.set_device(0)
        print('Running on GPU')
        class_weights = class_weights.cuda()
        model.cuda()
    else:
        print('Running on CPU')

    if args.loss == 'FocalLoss' and args.graph_model:
        # FocalLoss
        loss_f = FocalLoss(gamma=args.gamma, alpha=class_weights if args.class_weight else None)
    else:
        # NLLLoss
        loss_f = nn.NLLLoss(class_weights if args.class_weight else None) if args.graph_model \
            else MaskedNLLLoss(class_weights if args.class_weight else None)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    # scheduler_1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

    if args.dataset == 'MELD':
        train_loader, valid_loader, test_loader = get_MELD_loaders(data_path=args.data_dir,
                                                                   valid_rate=args.valid_rate,
                                                                   batch_size=batch_size,
                                                                   num_workers=0,
                                                                   args=args)
    elif args.dataset == 'IEMOCAP':
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(data_path=args.data_dir,
                                                                      valid_rate=args.valid_rate,
                                                                      batch_size=batch_size,
                                                                      num_workers=0,
                                                                      args = args)
    else:
        train_loader, valid_loader, test_loader = None, None, None
        print("There is no such dataset")

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []

    all_F1 = []
    for e in range(n_epochs):
        start_time = time.time()

        _, _, train_loss, train_acc, _, _, train_fscore, _, feature_a, feature_t, feature_fu,raw_list_audio,raw_list_text,preds,raw_preds = train_or_eval_graph_model(model=model,
                                                                                           loss_f=loss_f,
                                                                                           dataloader=train_loader,
                                                                                           epoch=e,
                                                                                           train_flag=True,
                                                                                           optimizer=optimizer,
                                                                                           cuda_flag=cuda_flag,
                                                                                           target_names=target_names)

        _, _, valid_loss, valid_acc, _, _, valid_fscore, _, _,_,_,_,_,_,_ = train_or_eval_graph_model(model=model,
                                                                                           loss_f=loss_f,
                                                                                           dataloader=valid_loader,
                                                                                           epoch=e,
                                                                                           cuda_flag=cuda_flag,
                                                                                           target_names=target_names)
        all_each, all_acc, test_loss, test_acc, test_label, test_pred, test_fscore, _,_,_,_,_,_,_,_  = train_or_eval_graph_model(model=model,
                                                                                                                      loss_f=loss_f,
                                                                                                                      dataloader=test_loader,
                                                                                                                      epoch=e,
                                                                                                                      cuda_flag=cuda_flag,
                                                                                                                      target_names=target_names)

        all_fscore.append(test_fscore)
        if max(all_fscore) == test_fscore:
            import os

            save_dir = args.save_model_dir
            if not os.path.isdir(save_dir): os.makedirs(save_dir)
            torch.save(model, os.path.join(save_dir, 'model_' + str(e) + '.pkl'))


        if args.tensorboard:
            writer.add_scalar('test: accuracy', test_acc, e)
            writer.add_scalar('test: fscore', test_fscore, e)
            writer.add_scalar('train: accuracy', train_acc, e)
            writer.add_scalar('train: fscore', train_fscore, e)
            writer.add_scalar('test: loss', test_loss, e)

        # print("第%d个epoch的学习率：%f" % (e, optimizer.param_groups[0]['lr']))

        print(
            'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
                format(e, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore,
                       round(time.time() - start_time, 2)))

        print(all_each)
        print(all_acc)
    if args.tensorboard:
        writer.close()

    print('Test performance..')
    print('F-Score:', max(all_fscore))
    print('Epoch:', all_fscore.index(max(all_fscore)))

