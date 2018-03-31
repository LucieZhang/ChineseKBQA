import time
# from preprocessing import entity_linking
from preprocessing import mention2id
from preprocessing import loder

kb_file_path = "E:/Lucy/Dissertation/code/ChineseKBQA/Data/nlpcc-iccpol-2016.kbqa.kb"
kb = loder.KnowledgeBase(kb_file_path)
kb.load_knowledge_base()

midpath = "E:/Lucy/Dissertation/code/ChineseKBQA/Data/nlpcc-iccpol-2016.kbqa.kb.mention2id"
mid = mention2id.MentionID()
mid.load_mention_2_id(midpath)


class NegativeSampling:
    def __init__(self, inpath, outpath, encode, men2idpath):
        self.inpath = inpath
        self.outpath = outpath
        self.encode = encode
        self.midpath = men2idpath

    def negative_sampling(self):
        with open(self.inpath, encoding=self.encode) as fi:
            # i = qIDstart
            timeStart = time.time()
            fo = open(self.outpath, 'w', encoding='utf8')
            fo.close()
            listQ = [] # query list
            qa_index = 1

            fo = open(self.outpath, 'w', encoding='utf-8')
            for line in fi:
                if line[1] == 'q':
                    same_query = line[line.index('\t') + 1:]

                if line[1] == 't':  # 问题对应的三元组对
                    same_triple = line[line.index('\t') + 1:]
                    sub = line[line.index('\t') + 1:line.index(' |||')].strip()  # 提取triple中的主语
                    qNSub = line[line.index(' ||| ') + 5:]  # 提取triple中的谓词和答案？
                    right_pre = qNSub[:qNSub.index(' |||')]  # 提取谓词
                    # candidate_ids = mid.find_id_set(sub)
                    # kb.show_entity("show_entity result:" + sub)
                    candidate = kb.get_all_predicates(sub)
                    if candidate and qa_index <= 50:
                        fo.write('<question id=' + str(qa_index) + '>\t' + same_query.lower())
                        fo.write('<triple id=' + str(qa_index) + '>\t' + same_triple.lower())
                        fo.write('<lable id=' + str(qa_index) + '>\t' + '1')
                        fo.write('\n==================================================\n')
                        qa_index += 1
                        for pre, ans in candidate.items():
                            if qa_index % 10000 == 0:
                                print('same_query: ' + same_query)
                                print("candidate predicates and answers are:" + pre + ans)
                            if pre != right_pre:
                                fo.write('<question id=' + str(qa_index) + '>\t' + same_query.lower())
                                fo.write('<triple id=' + str(qa_index) + '>\t' + sub + ' ||| ' + pre + ' ||| ' + ans + '\n')
                                fo.write('<lable id=' + str(qa_index) + '>\t' + '0')
                                fo.write('\n==================================================\n')
                                qa_index += 1

                    candidate.clear()

                    # for char in sub + pre:
                    #     if char not in countChar:
                    #         countChar[char] = -1
                    #     else:
                    #         countChar[char] = countChar[char] - 1
                # if line[1] == 't':
                #   listQ.append(line[line.index('\t') + 1:].strip())
            fo.close()
            print('Negative sampling finished!')


if __name__ == '__main__':
    inpath = "E:/Lucy/Dissertation/code/ChineseKBQA/Data/nlpcc-iccpol-2016.kbqa.training-data"
    outpath = "E:/Lucy/Dissertation/code/ChineseKBQA/Data/negative-sampling2.training-data"
    men2idpath = "E:/Lucy/Dissertation/code/ChineseKBQA/Data/nlpcc-iccpol-2016.kbqa.kb.mention2id"
    nega = NegativeSampling(inpath, outpath, 'utf-8', men2idpath)
    nega.negative_sampling()
