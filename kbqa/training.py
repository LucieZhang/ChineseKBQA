from kbqa import model
from preprocessing import loder
from preprocessing import query
from preprocessing import entity_linking

kb_file = "E:/Lucy/Dissertation/code/ChineseKBQA/Data/nlpcc-kb-small.txt"
mention2id_file = ""
train_file = ""
epoch_num = 5

kb = loder.KnowledgeBase(kb_file)
kb.load_knowledge_base()

question = query.QueryList()
question.read_query_file(train_file)
men2id = entity_linking.MentionExtractor()


answer_select = model.AnswerSelection()

# for epoch in range(epoch_num):# clear gradients