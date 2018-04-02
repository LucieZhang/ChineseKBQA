import torch

from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

from kbqa import data
from kbqa import DataServer
from kbqa import model
from preprocessing import loder
from preprocessing import query
from preprocessing import entity_linking

'''
kb_file = "E:/Lucy/Dissertation/code/ChineseKBQA/Data/nlpcc-kb-small.txt"
mention2id_file = ""
train_file = ""

# 读取知识库信息，训练阶段暂不需要
kb = loder.KnowledgeBase(kb_file)
kb.load_knowledge_base()

question = query.QueryList()
question.read_query_file(train_file)
men2id = entity_linking.MentionExtractor()
'''

qa_path = ''
save_path = '../network-parameters'

word_embedding_dim = 300
char_embedding_dim = 200
hidden_dim_word = 50
hidden_dim_char = 50
vocab_size
char_size
bs = 16
epoch_num = 5
learning_rate = 0.001

target_vocab, qa_data = data.load_data(opt, qa_path, 'training')
# train_data, valid_data, train_labels, valid_labels = train_test_split(qa_data[0], qa_data[1],
#                                                                      test_size=0.3, random_state=0)

network = model.AnswerSelection(word_embedding_dim, char_embedding_dim, hidden_dim_word,
                                hidden_dim_char, vocab_size, char_size, bs)

loss_function = nn.HingeEmbeddingLoss()
optimizer = optim.SGD(lr=learning_rate)

for epoch in range(epoch_num):  # clear gradients

    train_loader = DataServer([qa_data[0], qa_data[1]], vocab, opt)

    for i, data in enumerate(train_loader):
        qw, qc, tw, tc, labels = data
        optimizer.zero_grad()
        outputs = network(qw, qc, tw, tc)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch+1, epoch_num, i+1, len(qa_data)//bs, loss.data[0]))  # 检查len(qa_data)

torch.save(network.state_dict(), save_path)
