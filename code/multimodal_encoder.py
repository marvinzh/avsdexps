#!/usr/bin/env python
"""Multimodal sequence encoder 
   Copyright 2016 Mitsubishi Electric Research Labs
"""

import math
import numpy as np
import six
import torch
import torch.nn as nn
import torch.nn.functional as F


class MMEncoder(nn.Module):

    def __init__(self, in_size, out_size, enc_psize=[], enc_hsize=[], att_size=100,
                 state_size=100, device="cuda:0", enc_layers=[2,2], mm_att_size=100):
        if len(enc_psize)==0:
            enc_psize = in_size
        if len(enc_hsize)==0:
            enc_hsize = [0] * len(in_size)

        # make links
        super(MMEncoder, self).__init__()
        # memorize sizes
        self.n_inputs = len(in_size)
        self.in_size = in_size
        self.out_size = out_size
        self.enc_psize = enc_psize
        self.enc_hsize = enc_hsize
        self.enc_layers = enc_layers
        self.att_size = att_size
        self.state_size = state_size
        self.mm_att_size = mm_att_size
        # encoder
        self.f_lstms = nn.ModuleList()
        self.b_lstms = nn.ModuleList()
        self.emb_x = nn.ModuleList()
        self.device= torch.device(device)
        for m in six.moves.range(len(in_size)):
            self.emb_x.append(nn.Linear(self.in_size[m], self.enc_psize[m]))

            if enc_hsize[m] > 0:
                # create module for stacked bi-LSTM
                self.f_lstms.append(torch.nn.ModuleList())
                self.b_lstms.append(torch.nn.ModuleList())
                # create stacked bi-LSTM for current modality m
                for layer in range(self.enc_layers[m]):
                    if layer == 0:
                        self.f_lstms[m].append(nn.LSTMCell(enc_psize[m], enc_hsize[m]).to(self.device))
                        self.b_lstms[m].append(nn.LSTMCell(enc_psize[m], enc_hsize[m]).to(self.device))
                    else:
                        self.f_lstms[m].append(nn.LSTMCell(enc_hsize[m], enc_hsize[m]).to(self.device))
                        self.b_lstms[m].append(nn.LSTMCell(enc_hsize[m], enc_hsize[m]).to(self.device))

                # self.b_lstms.append(nn.LSTMCell(enc_psize[m], enc_hsize[m]).to(self.device))
        # temporal attention
        self.atV = nn.ModuleList()
        self.atW = nn.ModuleList()
        self.atw = nn.ModuleList()
        self.lgd = nn.ModuleList()
        for m in six.moves.range(len(in_size)):
            enc_hsize_ = 2 * enc_hsize[m] if enc_hsize[m] > 0 else enc_psize[m]
            self.atV.append(nn.Linear(enc_hsize_, att_size))
            self.atW.append(nn.Linear(state_size, att_size))
            self.atw.append(nn.Linear(att_size, 1))
            self.lgd.append(nn.Linear(enc_hsize_, out_size))


        # multimodal attention
        self.mm_atts = nn.ModuleList()
        self.qest_att = nn.Linear(128, self.mm_att_size)
        self.mm_att_w = nn.Linear(self.mm_att_size, 1, bias=False)
        for m in six.moves.range(len(in_size)):
            enc_hsize_ = 2 * enc_hsize[m] if enc_hsize[m] > 0 else enc_psize[m]
            self.mm_atts.append(nn.Linear(enc_hsize_, self.mm_att_size))
    
        

    # Make an initial state
    def make_initial_state(self, hiddensize):
        return (
            # initial hidden state
            torch.zeros(self.bsize, hiddensize, dtype=torch.float).to(self.device),
            # initial cell state
            torch.zeros(self.bsize, hiddensize, dtype=torch.float).to(self.device),
        )

    # Encoder functions
    def embed_x(self, x_data, m):
        x0 = [x_data[i]
              for i in six.moves.range(len(x_data))]
        return self.emb_x[m](torch.cat(x0, 0).cuda().float())

    # Encoder main
    def encode(self, x):
        h1 = [None] * self.n_inputs
        for m in six.moves.range(self.n_inputs):
            if self.enc_hsize[m] > 0:
                # embedding
                seqlen = len(x[m])
                h0 = self.embed_x(x[m], m)
                # forward path
                fh1 = torch.split(
                        F.dropout(h0, training=self.train), 
                        self.bsize, dim=0)

                hs, cs= self.make_initial_state(self.enc_hsize[m])
                # extend initial hidden state and cell state for stacked LSTM
                hs = [[hs]] * len(self.b_lstms[m])
                cs = [[cs]] * len(self.b_lstms[m])
                h1f = []

                for h in fh1:
                    for level in range(len(self.f_lstms[m])):
                        if level==0:
                            hs_temp, cs_temp = self.f_lstms[m][level](
                                h, 
                                (hs[level][-1], cs[level][-1])
                                )
                        else:
                            hs_temp, cs_temp = self.f_lstms[m][level](
                                hs[level-1][-1],
                                (hs[level][-1],cs[level][-1])
                                )
                        hs[level].append(hs_temp)
                        cs[level].append(cs_temp)
                    # fstate = self.f_lstms[m](h,fstate)
                    h1f.append(hs[-1][-1])

                # backward path
                bh1 = torch.split(
                        F.dropout(h0, training=self.train),
                        self.bsize, dim=0)

                hs, cs = self.make_initial_state(self.enc_hsize[m])
                 # extend initial hidden state and cell state for stacked LSTM
                hs = [[hs]] * len(self.b_lstms[m])
                cs = [[cs]] * len(self.b_lstms[m])
                h1b = []
                for h in reversed(bh1):
                    for level in range(len(self.b_lstms[m])):
                        if level == 0:
                            hs_temp, cs_temp = self.b_lstms[m][level](
                                h,
                                (hs[level][-1], cs[level][-1])
                                )
                        else:
                            hs_temp, cs_temp = self.b_lstms[m][level](
                                hs[level-1][-1],
                                (hs[level][-1], cs[level][-1])
                                )
                        hs[level].append(hs_temp)
                        cs[level].append(cs_temp)

                    # bstate = self.b_lstms[m](h, bstate)
                    h1b.insert(0, hs[-1][-1])

                # concatenation
                h1[m] = torch.cat([torch.cat((f, b), 1)
                                   for f, b in six.moves.zip(h1f, h1b)], 0)
            else:
                # embedding only
                h1[m] = torch.tanh(self.embed_x(x[m], m))
        return h1

    # Attention
    def attention(self, h, vh, s):
        c = [None] * self.n_inputs

        for m in six.moves.range(self.n_inputs):
            bsize = self.bsize
            seqlen = h[m].data.shape[0] / bsize
            csize = h[m].data.shape[1]
            asize = self.att_size

            ws = self.atW[m](s)
            vh_m = vh[m].view(seqlen, bsize, asize)
            e1 = vh_m + ws.expand_as(vh_m)
            e1 = e1.view(seqlen * bsize, asize)
            e = torch.exp(self.atw[m](torch.tanh(e1)))
            e = e.view(seqlen, bsize)
            esum = e.sum(0)
            e = e / esum.expand_as(e)
            h_m = h[m].view(seqlen, bsize, csize)
            h_m = h_m.permute(2,0,1)
            c_m = h_m * e.expand_as(h_m)
            c_m = c_m.permute(1,2,0)
            c[m] = c_m.mean(0)
        return c

    def mm_attention(self, g_q, c):
        wg= self.qest_att(g_q)
        vs=[]
        for i in range(self.n_inputs):
             vs.append(
                 self.mm_atts[i](c[i]) + wg
                 )

        # each elems in vs (B, atten_size)
        for i in range(self.n_inputs):
            vs[i] = self.mm_att_w(torch.tanh(vs[i]))
        
        # each elems in vs (B, 1)
        vs = torch.cat(vs,dim=1)
        #  (B, # of modality)
        beta = torch.softmax(vs, dim=1)

        # (batchsize, #modality)
        return beta
    
    def att_modality_fusion(self, c, beta):
        assert beta.shape[1] == self.n_inputs
        attended = [None] * self.n_inputs

        beta = beta.permute(1,0)
        # beta: (# of modality, B)
        g = 0.
        for m in range(self.n_inputs):
            attended[m] = beta[m].view(-1,1) * c[m]
            g += self.lgd[m](attended[m])
        return g

    # Simple modality fusion
    def simple_modality_fusion(self, c, s):

        g = 0.
        for m in six.moves.range(self.n_inputs):
            g += self.lgd[m](F.dropout(c[m]))
        return g

    # forward propagation routine
    def __call__(self, s, x, train=True):
        '''multimodal encoder main
        
        Arguments:
            s {[type]} -- question encoding
            x {[type]} -- raw multi-modal feature
        
        Keyword Arguments:
            train {bool} -- [description] (default: {True})
        
        Returns:
            [type] -- [description]
        '''

        self.bsize = x[0][0].shape[0]
        
        h1 = self.encode(x)
        vh1 = [self.atV[m](h1[m]) for m in six.moves.range(self.n_inputs)]

        # attention
        c = self.attention(h1, vh1, s)

        beta = self.mm_attention(s, c)
        g = self.att_modality_fusion(c, beta)
        # g = self.simple_modality_fusion(c, s)
        
        return torch.tanh(g)

