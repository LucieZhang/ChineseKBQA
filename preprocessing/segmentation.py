'''
分词 + 命名实体识别
1.pynlp
'''


import os
LTP_DATA_DIR = '/path/to/your/ltp_data'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`

#from pyltp import Segmentor
# segmentor = Segmentor()  # 初始化实例
# segmentor.load(cws_model_path)  # 加载模型
# words = segmentor.segment('元芳你怎么看')  # 分词
# print '\t'.join(words)
# segmentor.release()  # 释放模型

import thulac
'''
thulac对实体分词效果不太好，filt=true时容易过滤掉关键信息，有数字、字母时准确度下降
'''
thu1 = thulac.thulac(seg_only=True, model_path="E:\Lucy\Dissertation\code\THULAC-Python-master\models")  #只进行分词，不进行词性标注
thu1.cut_f("../Data/seg_test.txt", "../Data/2016kbqa_seg2.txt")  #对input.txt文件内容进行分词，输出到output.txt
#nlpcc-iccpol-2016.kbqa.kb
#thu2 = thulac.thulac(seg_only=True, model_path="E:\Lucy\Dissertation\code\THULAC-Python-master\models")  #设置模式为行分词模式
#a = thu2.cut("我爱北京天安门")
#print(a)
