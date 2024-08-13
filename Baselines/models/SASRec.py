import torch
from .layers import EmbeddingLayer, MultiLayerPerceptron
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.utility import *


class SASRecModel(torch.nn.Module):
    def __init__(self, hidden_size, item_num, state_size, device):
        super().__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = item_num
        self.device = device

        self.state_embeddings = nn.Embedding(self.item_num+1, self.hidden_size)
        self.pos_embeddings = nn.Embedding(self.state_size, self.hidden_size)

        self.transformer = TransformerEncoder(TransformerEncoderLayer(d_model=self.hidden_size, nhead=1), num_layers=1)

        # self.fc_1 = nn.Linear(self.hidden_size, self.item_num)
        # self.fc_2 = nn.Linear(self.hidden_size, self.item_num)

        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs, len_state, is_training):
        input_emb = self.state_embeddings(inputs)
        pos_emb = self.pos_embeddings(torch.arange(self.state_size).unsqueeze(0).to(self.device))

        seq = input_emb + pos_emb

        mask = (inputs != self.item_num).unsqueeze(-1).float()

        if is_training:
            seq = self.dropout(seq)

        seq = seq * mask

        seq = self.transformer(seq)

        state_hidden = extract_axis_1(seq, len_state-1, self.device)

        # output = [self.fc_1(state_hidden), self.fc_2(state_hidden)]

        return state_hidden


class SASRecModel_2(torch.nn.Module):

    def __init__(self, hidden_size, item_num, state_size, device):
        super().__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = item_num
        self.device = device

        self.state_embeddings = nn.Embedding(self.item_num+1, self.hidden_size)
        self.pos_embeddings = nn.Embedding(self.state_size, self.hidden_size)

        self.transformer = TransformerEncoder(TransformerEncoderLayer(d_model=self.hidden_size, nhead=1), num_layers=1)

        self.fc_1 = nn.Linear(self.hidden_size, self.item_num)
        self.fc_2 = nn.Linear(self.hidden_size, self.item_num)

        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs, len_state, is_training):
        input_emb = self.state_embeddings(inputs)
        pos_emb = self.pos_embeddings(torch.arange(self.state_size).unsqueeze(0).to(self.device))

        seq = input_emb + pos_emb

        mask = (inputs != self.item_num).unsqueeze(-1).float()

        if is_training:
            seq = self.dropout(seq)

        seq = seq * mask

        seq = self.transformer(seq)

        state_hidden = extract_axis_1(seq, len_state-1, self.device)

        output = [self.fc_1(state_hidden), self.fc_2(state_hidden)]

        return output