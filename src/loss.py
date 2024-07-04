import torch
import torch.nn as nn
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):#log_prob(1491,6),label(1491,)
        if input.dim()>2:
            # input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            # input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            # input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
            input = input.view(-1, 2)
        target = target.view(-1,1)   #view功能与reshape类似，第二个维度为1，第一个自动计算，此处是将lable(1491)--->(1491,1)方便与log_prob计算

        logpt = input

        logpt = logpt.gather(1,target)#logpt(1491,)是log_prob中每一个utterance预测的6个值中对应真实label下标的值
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        '''#Variable()方法用于将张量（tensor）封装成可求导的对象，在反向传播过程中自动求导,
        由于logpt是对预测标签做log结果，所以想要得到原始pt，需要做指数运算，即logpt.data.exp()'''

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        '''对应公式-α(1-pt)^γ*logpt'''
        return loss.mean() if self.size_average else loss.sum()



class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1, 1)
        if type(self.weight) == type(None):
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) \
                   / torch.sum(self.weight[target] * mask_.squeeze())
        return loss


class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len
        target -> batch*seq_len
        mask -> batch*seq_len
        """
        loss = self.loss(pred * mask, target) / torch.sum(mask)
        return loss