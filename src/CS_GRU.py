import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.autograd import Variable


class CS_GRU(nn.Module):
    def __init__(self, indim, hidim, outdim):
        super(CS_GRU, self).__init__()
        self.indim = indim
        self.hidim = hidim
        self.outdim = outdim
        self.W_zh, self.W_zx, self.W_zy, self.W_zq, self.b_z = self.get_five_parameters()
        self.W_rh, self.W_rx, self.W_ry, self.W_rq, self.b_r = self.get_five_parameters()
        self.W_hh, self.W_hx, self.W_hy, self.W_hq, self.b_h = self.get_five_parameters()
        self.Linear = nn.Linear(hidim, outdim)  # 全连接层做输出
        self.reset()

    def forward(self, input, Y, Q, state):
        input = input.type(torch.float32)
        Y = Y.type(torch.float32)
        Q = Q.type(torch.float32)
        if torch.cuda.is_available():
            input = input.cuda()
            Y = Y.cuda()
            Q = Q.cuda()
        T = []
        h = state
        h = h.cuda()
        for x in input:
            z = F.sigmoid(h @ self.W_zh + x @ self.W_zx + Y @ self.W_zy + Q @ self.W_zq + self.b_z)
            r = F.sigmoid(h @ self.W_rh + x @ self.W_rx + Y @ self.W_ry + Q @ self.W_rq + self.b_r)
            ht = F.tanh((h * r) @ self.W_hh + x @ self.W_hx + Y @ self.W_hy + Q @ self.W_hq + self.b_h)
            h = (1 - z) * h + z * ht
            t = self.Linear(h)
            T.append(t)
        return torch.cat(T, dim=0), h

    def get_three_parameters(self):
        indim, hidim, outdim = self.indim, self.hidim, self.outdim
        return nn.Parameter(torch.randn(hidim, hidim)*0.01), \
               nn.Parameter(torch.randn(hidim, hidim)*0.01), \
               nn.Parameter(torch.randn(hidim, hidim)*0.01), \
               nn.Parameter(torch.randn(hidim, hidim)*0.01), \
               nn.Parameter(torch.FloatTensor(hidim))

    def reset(self):
        stdv = 1.0 / math.sqrt(self.hidim)
        for param in self.parameters():
            nn.init.uniform_(param, -stdv, stdv)

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.y2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.q2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()
        # self.lnx = nn.LayerNorm(input_size*3)
        # self.lnh = nn.LayerNorm(hidden_size*3)
        # self.lny = nn.LayerNorm(input_size*3)
        # self.lnq = nn.LayerNorm(input_size*3)

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, yi, qi,  hx=None):
        '''Inputs:
              input: of shape (batch_size, input_size)
              hx: of shape (batch_size, hidden_size)
        Output:
              hy: of shape (batch_size, hidden_size)'''
        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.hidden_size))

        x_t = self.x2h(input)
        h_t = self.h2h(hx)
        y_t = self.y2h(yi)
        q_t = self.q2h(qi)

        x_reset, x_upd, x_new = x_t.chunk(3, 1)
        h_reset, h_upd, h_new = h_t.chunk(3, 1)
        y_reset, y_upd, y_new = y_t.chunk(3, 1)
        q_reset, q_upd, q_new = q_t.chunk(3, 1)

        reset_gate = torch.sigmoid(x_reset + h_reset + q_reset + y_reset)
        update_gate = torch.sigmoid(x_upd + h_upd + q_upd + y_upd)
        new_gate = torch.tanh(x_new + (reset_gate * h_new) + q_new + y_new)

        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy

class CS_GRU_v1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, output_size):
        super(CS_GRU_v1, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size
        self.rnn_cell_list = nn.ModuleList()
        self.rnn_cell_list.append(GRUCell(self.input_size,
                                          self.hidden_size,
                                          self.bias))
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(GRUCell(self.hidden_size,
                                              self.hidden_size,
                                              self.bias))
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, Y, Q, hx=None):
        '''
        Input of shape (batch_size, seqence length, input_size)
        Output of shape (batch_size, output_size)'''
        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda())
            else:
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size))
        else:
             h0 = hx

        outs = []
        hidden = list()
        for layer in range(self.num_layers):
            hidden.append(h0[layer, :, :])

        for t in range(input.size(1)):
            for layer in range(self.num_layers):
                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](input[:, t, :], Y[:, t, :], Q[:, t, :], hidden[layer])
                else:
                    hidden_l = self.rnn_cell_list[layer](hidden[layer - 1], Y[:, t, :], Q[:, t, :], hidden[layer])
                hidden[layer] = hidden_l

            outs.append(hidden_l.unsqueeze(1))

        # Take only last time step. Modify for seq to seq
        out = torch.cat(outs, dim=1)

        out = self.fc(out)

        return out

class GRUCell_v2(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell_v2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.y2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.q2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.s2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()
        # self.lnx = nn.LayerNorm(input_size*3)
        # self.lnh = nn.LayerNorm(hidden_size*3)
        # self.lny = nn.LayerNorm(input_size*3)
        # self.lnq = nn.LayerNorm(input_size*3)

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, yi, qi, si, hx=None):
        '''Inputs:
              input: of shape (batch_size, input_size)
              hx: of shape (batch_size, hidden_size)
        Output:
              hy: of shape (batch_size, hidden_size)'''
        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.hidden_size))

        x_t = self.x2h(input)
        h_t = self.h2h(hx)
        y_t = self.y2h(yi)
        q_t = self.q2h(qi)
        s_t = self.s2h(si)

        x_reset, x_upd, x_new = x_t.chunk(3, 1)
        h_reset, h_upd, h_new = h_t.chunk(3, 1)
        y_reset, y_upd, y_new = y_t.chunk(3, 1)
        q_reset, q_upd, q_new = q_t.chunk(3, 1)
        s_reset, s_upd, s_new = s_t.chunk(3, 1)

        reset_gate = torch.sigmoid(x_reset + h_reset + q_reset + y_reset + s_reset)
        update_gate = torch.sigmoid(x_upd + h_upd + q_upd + y_upd + s_upd)
        new_gate = torch.tanh(x_new + (reset_gate * h_new) + q_new + y_new + s_new)

        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy

class CS_GRU_v2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, output_size):
        super(CS_GRU_v2, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size
        self.rnn_cell_list = nn.ModuleList()
        self.rnn_cell_list.append(GRUCell_v2(self.input_size,
                                          self.hidden_size,
                                          self.bias))
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(GRUCell_v2(self.hidden_size,
                                              self.hidden_size,
                                              self.bias))
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, Y, Q, S, hx=None):
        '''
        Input of shape (batch_size, seqence length, input_size)
        Output of shape (batch_size, output_size)'''
        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda())
            else:
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size))
        else:
             h0 = hx

        outs = []
        hidden = list()
        for layer in range(self.num_layers):
            hidden.append(h0[layer, :, :])

        for t in range(input.size(1)):
            for layer in range(self.num_layers):
                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](input[:, t, :], Y[:, t, :], Q[:, t, :], S[:, t, :], hidden[layer])
                else:
                    hidden_l = self.rnn_cell_list[layer](hidden[layer - 1], Y[:, t, :], Q[:, t, :], S[:, t, :], hidden[layer])
                hidden[layer] = hidden_l

            outs.append(hidden_l.unsqueeze(1))

        # Take only last time step. Modify for seq to seq
        out = torch.cat(outs, dim=1)

        out = self.fc(out)

        return out


class GRUCell_v3(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell_v3, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.y2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()
        # self.lnx = nn.LayerNorm(input_size*3)
        # self.lnh = nn.LayerNorm(hidden_size*3)
        # self.lny = nn.LayerNorm(input_size*3)
        # self.lnq = nn.LayerNorm(input_size*3)

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, yi, hx=None):
        '''Inputs:
              input: of shape (batch_size, input_size)
              hx: of shape (batch_size, hidden_size)
        Output:
              hy: of shape (batch_size, hidden_size)'''
        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.hidden_size))

        x_t = self.x2h(input)
        h_t = self.h2h(hx)
        y_t = self.y2h(yi)

        x_reset, x_upd, x_new = x_t.chunk(3, 1)
        h_reset, h_upd, h_new = h_t.chunk(3, 1)
        y_reset, y_upd, y_new = y_t.chunk(3, 1)

        reset_gate = torch.sigmoid(x_reset + h_reset + y_reset)
        update_gate = torch.sigmoid(x_upd + h_upd + y_upd)
        new_gate = torch.tanh(x_new + (reset_gate * h_new) + y_new)

        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy