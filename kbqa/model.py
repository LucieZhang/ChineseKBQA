import codecs

import fastai
import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import pickle

import numpy as np
from torch.nn.utils import clip_grad_norm

# sent_len = 30  # need padding


class AnswerSelection(nn.Module):
    def __init__(self, word_embedding_dim, char_embedding_dim, hidden_dim_word, hidden_dim_char,
                 vocab_size, char_size, bs):
        super(AnswerSelection, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)
        self.char_embeddings = nn.Embedding(char_size, char_embedding_dim)
        self.lstm_word = nn.LSTM(word_embedding_dim, hidden_dim_word, bidirectional=True)
        self.lstm_char = nn.LSTM(char_embedding_dim, hidden_dim_char, bidirectional=True)
        self.bs = bs
        self.hidden_dim_word = hidden_dim_word
        self.hidden_dim_char = hidden_dim_char
        self.hidden_qword = self.init_hidden_word()
        self.hidden_qchar = self.init_hidden_char()
        self.hidden_tword = self.init_hidden_word()
        self.hidden_tchar = self.init_hidden_char()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def init_hidden_word(self):
        return autograd.Variable(torch.randn(1, self.bs, self.hidden_dim_word))

    def init_hidden_char(self):
        return autograd.Variable(torch.randn(1, self.bs, self.hidden_dim_char))

    def forward(self, question_wordseq, question_charseq, triple_wordseq, triple_charseq):
        qword_embeds = self.word_embeddings(question_wordseq)
        qchar_embeds = self.char_embeddings(question_charseq)
        tword_embeds = self.word_embeddings(triple_wordseq)
        tchar_embeds = self.char_embeddings(triple_charseq)

        lstm_out_qword, self.hidden_qword = self.lstm_word(qword_embeds.view(1, self.bs, -1), self.hidden_qword)
        lstm_out_qchar, self.hidden_qchar = self.lstm_char(qchar_embeds.view(1, self.bs, -1), self.hidden_qchar)
        lstm_out_tword, self.hidden_tword = self.lstm_word(tword_embeds.view(1, self.bs, -1), self.hidden_tword)
        lstm_out_tchar, self.hidden_tchar = self.lstm_char(tchar_embeds.view(1, self.bs, -1), self.hidden_tchar)

        q_representation = torch.cat((lstm_out_qword[-1], lstm_out_qchar[-1]), 2)  # Need to check the second param
        t_representation = torch.cat((lstm_out_tword[-1], lstm_out_tchar[-1]), 2)
        score = self.cos(q_representation, t_representation)

        # 在score上加一层non-linearity

        return score
