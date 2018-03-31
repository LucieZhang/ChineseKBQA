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

sent_len = 30  # need padding


class LSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, char_embedding_dim, hidden_dim, vocab_size, char_size, sent_dim, bs):
        super(LSTMEncoder, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Chinese character level embedding
        self.char_embeddings = nn.Embedding(char_size, char_embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        # self.l_out = nn.Linear(hidden_dim, sent_dim) # sentVector
        self.bs = bs
        self.hidden_dim = hidden_dim
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, self.bs, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, self.bs, self.hidden_dim)))

    def forward(self, *input_sentence):
        embeds = self.word_embeddings(input_sentence)
        # h = autograd.Variable(torch.zero(1, self.bs, self.hidden_dim).cuda())
        lstm_out, self.hidden = self.lstm(embeds.view(sent_len, 1, -1), self.hidden)

        return lstm_out[-1]


class AnswerSelection(nn.Module):
    def __init__(self, vocab_size, opt, embedding_dim, hidden_dim, labels):
        super(AnswerSelection, self).__init__()
        self.opt = opt
        self.sent_a = self.sent_b = LSTMEncoder(embedding_dim, hidden_dim, vocab_size)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.score = 0
        self.loss = 0
        self.labels = labels

    def forward(self):
        out_a = self.sent_a()
        out_b = self.sent_b()
        self.score = self.cos(out_a, out_b)
        return self.score

    def get_loss(self):
        self.loss = self.loss_function(AnswerSelection.forward(), self.labels)

    def train(self, train_labels):
        self.labels = train_labels
        self.forward()
        self.sent_a.zero_grad()
        self.get_loss()
        self.loss.backward()

        #clip_grad_norm(self.)

        optimizer = optim.SGD(self.sent_a.parameters(), lr=0.1)  # to use fastai method
        optimizer.step()

    def test_step(self, test_batch_a, test_batch_b, test_labels):
        """ Performs a single test step. """
        # Get batches
        #self.batch_a = test_batch_a
        #self.batch_b = test_batch_b
        self.labels = test_labels

        # Get batch_size for current batch
        #self.batch_size = self.batch_a.size(1)

        svr_path = os.path.join(self.opt.save_dir, 'sim_svr.pkl')
        if os.path.exists(svr_path):
            # Correct predictions via trained SVR
            with open(svr_path, 'rb') as f:
                sim_svr = pickle.load(f)
            self.forward()
            self.score = autograd.Variable(torch.FloatTensor(sim_svr.predict(self.score.view(-1, 1).data.numpy())))

        else:
            self.forward()

        self.get_loss()


if __name__ == 'main':
    filename = ""
    fh = codecs.open(filename, 'r', encoding='utf-8')
    lines = fh.readlines()
    for i in range(0, len(lines)):
        query = lines[i].strip()
        ans_select = AnswerSelection()
