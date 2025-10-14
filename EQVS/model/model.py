"""
DATE: 15/07/2023
LAST CHANGE: 27/07/2023
AUTHOR: CHENG ZHANG

Model without modules
"""

import torch
import torch.nn as nn
import numpy as np
import math
from model import retention

class Model(nn.Module):
    def __init__(self, sample_length):
        super(Model, self).__init__()

        ################################################################################################################
        # This part informs the model about the sample lengths and defines the number of output channels used in the   #
        # model                                                                                                        #
        ################################################################################################################
        self.sample_length = sample_length
        self.input_frame = int(1000 * sample_length // 10)

        self.frames_in_group = 5
        self.group = int(self.input_frame/self.frames_in_group)

        self.embed_dim = 10

        self.gamma = 0.8
        self.pk_list = [0, 0, 0, 0, 1, 3, 1, 2, 3, 2, 3, 1, 2, 3, 3]
        self.D = torch.tensor([self.gamma ** x for x in self.pk_list]).cuda().repeat_interleave(self.embed_dim)

        # Representation table
        self.representation_table = nn.Embedding(8192, self.embed_dim)

        # conv
        self.DWC_lsp = nn.Conv2d(self.group, self.group, kernel_size=3, padding=1, bias=True)
        self.DWC_lsp_other = nn.Conv2d(self.group, self.group, kernel_size=3, padding=1, bias=True)

        self.conv_local = nn.Conv1d(self.input_frame, self.input_frame, kernel_size=1, bias=True)
        self.conv_global = nn.Conv1d(self.input_frame, self.input_frame, kernel_size=1, bias=True)

        # query
        # gain
        self.gain_q_w = nn.Linear(self.embed_dim * 4, self.embed_dim * 4)
        # code
        self.code_q_w = nn.Linear(self.embed_dim * 4, self.embed_dim * 4)
        # pitch
        self.pitch_q_w = nn.Linear(self.embed_dim * 3, self.embed_dim * 4)

        # key
        self.lsp_k_w = nn.Linear(self.embed_dim * 4, self.embed_dim * 4)

        # value
        # gain
        self.lsp_v_w = nn.Linear(self.embed_dim * 4, self.embed_dim * 4)


        # TODO 计算维度
        self.classifier = nn.Sequential(
            nn.Linear(self.input_frame*4*self.embed_dim*3, 1, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

    def forward(self, X):
        # size (4, 1000, 15)
        batch, seq, codeword = X.size()

        # Representation
        X = self.representation_table(X).view(batch, seq, -1)
        X = X * self.D
        # L0, 1 bit
        l0 = X[:, :, 0:1*self.embed_dim]
        # L1, 7 bits
        l1 = X[:, :, 1*self.embed_dim:2*self.embed_dim]
        # L2, 5 bits
        l2 = X[:, :, 2*self.embed_dim:3*self.embed_dim]
        # L3, 5 bits
        l3 = X[:, :, 3*self.embed_dim:4*self.embed_dim]
        # P1, 8 bits
        p1 = X[:, :, 4*self.embed_dim:5*self.embed_dim]
        # P0, 1 bit
        p0 = X[:, :, 5*self.embed_dim:6*self.embed_dim]
        # C1, 13 bits
        c1 = X[:, :, 6*self.embed_dim:7*self.embed_dim]
        # Si, 4 bits
        s1 = X[:, :, 7*self.embed_dim:8*self.embed_dim]
        # GA1, 3 bits
        ga1 = X[:, :, 8*self.embed_dim:9*self.embed_dim]
        # GB1, 4 bits
        gb1 = X[:, :, 9*self.embed_dim:10*self.embed_dim]
        # P2, 5 bits
        p2 = X[:, :, 10*self.embed_dim:11*self.embed_dim]
        # C2, 13 bits
        c2 = X[:, :, 11*self.embed_dim:12*self.embed_dim]
        # S2, 4 bits
        s2 = X[:, :, 12*self.embed_dim:13*self.embed_dim]
        # GA2, 3 bits
        ga2 = X[:, :, 13*self.embed_dim:14*self.embed_dim]
        # GB2, 4 bits
        gb2 = X[:, :, 14*self.embed_dim:15*self.embed_dim]

        # LSP
        lsp = torch.cat((l0, l1, l2, l3), -1)
        # Pitch
        pitch = torch.cat((p1, p0, p2), -1)
        # Code
        code = torch.cat((c1, s1, c2, s2), -1)
        # Gain
        gain = torch.cat((ga1, ga2, gb1, gb2), -1)

        lsp_reshape = lsp.reshape(batch, self.group, self.frames_in_group, -1)
        pitch_reshape = pitch.reshape(batch, self.group, self.frames_in_group, -1)
        code_reshape = code.reshape(batch, self.group, self.frames_in_group, -1)
        gain_reshape = gain.reshape(batch, self.group, self.frames_in_group, -1)

        conv_lsp = self.DWC_lsp(lsp_reshape).reshape(batch, seq, -1)
        conv_pitch = self.DWC_lsp_other(pitch_reshape).reshape(batch, seq, -1)
        conv_code = self.DWC_lsp_other(code_reshape).reshape(batch, seq, -1)
        conv_gain = self.DWC_lsp_other(gain_reshape).reshape(batch, seq, -1)

        conv_lsp = self.conv_local(conv_lsp)
        conv_pitch = self.conv_local(conv_pitch)
        conv_code = self.conv_local(conv_code)
        conv_gain = self.conv_local(conv_gain)

        # query
        gain_query = self.gain_q_w(conv_gain)
        pitch_query = self.pitch_q_w(conv_pitch)
        code_query = self.code_q_w(conv_code)

        # gain key and value
        lsp_key = self.lsp_k_w(conv_lsp)
        lsp_value = self.lsp_v_w(conv_lsp)

        lsp_gain_atten_out = gain_query @ lsp_key.permute(0, 2, 1) @ lsp_value
        lsp_pitch_atten_out = pitch_query @ lsp_key.permute(0, 2, 1) @ lsp_value
        lsp_code_atten_out = code_query @ lsp_key.permute(0, 2, 1) @ lsp_value

        lsp_gain_atten_out = lsp_gain_atten_out.view(batch, seq, -1) + lsp
        lsp_pitch_atten_out = lsp_pitch_atten_out.view(batch, seq, -1) + lsp
        lsp_code_atten_out = lsp_code_atten_out.view(batch, seq, -1) + lsp

        atten_out = torch.cat((lsp_gain_atten_out, lsp_pitch_atten_out, lsp_code_atten_out), -1)

        global_feature = self.conv_global(atten_out).view(batch, -1)

        out = self.classifier(global_feature)

        return out