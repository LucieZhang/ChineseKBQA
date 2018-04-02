import codecs
import re
import time


'''
    def load_knowledge_base(self, encode='utf-8'):
        print("start loading kb")
        with open(self.kb_file_path, 'r', encoding=encode) as fi:
            prePattern = re.compile(r'[·•\-\s]|(\[[0-9]*\])')

            lines = fi.readlines()

            # kb_dict = {}
            num_tri = 0
            for line in range(0, len(lines)):
                num_tri += 1
                # print("loading knowledge base")
                print('exporting the ' + str(num_tri) + ' triple', end='\r', flush=True)
                entityStr = line[:line.index(' |||')].strip()

                tmp = line[line.index('||| ') + 4:]
                relationStr = tmp[:tmp.index(' |||')].strip()
                relationStr, num = prePattern.subn('', relationStr)
                objectStr = tmp[tmp.index('||| ') + 4:].strip()
                if relationStr == objectStr:  # delete the triple if the predicate is the same as object
                    continue
                if entityStr not in self.knowledge_about:
                    newEntityDic = {relationStr: objectStr}
                    self.knowledge_about[entityStr] = []
                    self.knowledge_about[entityStr].append(newEntityDic)
                else:
                    self.knowledge_about[entityStr][len(self.knowledge_about[entityStr]) - 1][relationStr] = objectStr

        return self.knowledge_about
'''


class KnowledgeBase:
    def __init__(self, kb_file_path):
        self.knowledge_about = {}
        self.kb_file_path = kb_file_path
        self.candidate = {}
        self.total_line_number = 0
        self.skip = 0

    # def read_in_chunks(self, chunk_size=1024 * 1024):
    #     file_object = open(self.kb_file_path, 'r', encoding='utf-8')
    #     while True:
    #         chunk_data = file_object.read(chunk_size)
    #         if not chunk_data:
    #             break
    #         yield chunk_data

    def process_line(self, line):
        try:
            # for line in chunk:  # enumerate(fh.readline()):
            # if line_no > 101840:
            #     print 'line content:', line.rstrip()
            self.total_line_number += 1
            if self.total_line_number % 1000000 == 0:
                print('Processed', self.total_line_number, 'lines')
            try:
                entity1, predicate, entity2 = line.rstrip().split(' ||| ')
                # print(entity1, predicate, entity2)
            except ValueError:
                self.skip += 1
                # continue
            try:
                self.knowledge_about[entity1].append((predicate, entity2))
            except KeyError:
                self.knowledge_about[entity1] = [(predicate, entity2)]

        except MemoryError:
            print("stuck at line number: " + self.total_line_number)

    def load_knowledge_base(self):
        # Typically, it takes ~130s to load this file.
        t1 = time.time()
        print("start loading kb")
        with open(self.kb_file_path, 'r', encoding='utf-8') as fh:
        # for chunk in self.read_in_chunks():
            for line in fh:
                if self.total_line_number <= 1000000:
                    self.process_line(line)

        self.total_line_number -= 1
        print('total line number:', self.total_line_number)
        print('skipped:', self.skip)

        for entity in self.knowledge_about.keys():
            self.knowledge_about[entity] = set(self.knowledge_about[entity])
        t2 = time.time()
        print('Loading knowledge base consumed', t2 - t1, 'seconds')

    def show(self, first_n=20):
        for i, entity in enumerate(self.knowledge_about.keys()):
            # print entity, len(entity)
            print(entity)
            for predicate, entity2 in self.knowledge_about[entity]:
                # print predicate, len(predicate), type(predicate), entity2, len(entity2), type(entity2)
                print('\t', predicate, entity2)
            if i > first_n:
                break

    def show_entity(self, entity):
        if entity in self.knowledge_about:
            print(entity, 'in knowledge base! attributes:')
            for item in self.knowledge_about[entity]:
                predicate, entity2 = item
                print(predicate, entity2)
            pass
        else:
            print('Unfortunately,' + entity + 'not found!')

    def get_all_predicates(self, entity):
        self.candidate.clear()
        if entity in self.knowledge_about:
            for i in self.knowledge_about[entity]:
                predicate, answer = i
                self.candidate[predicate] = answer
        else:
            print(entity + 'not found!')
        return self.candidate


# 将所有的 mention2id 可能的组合存储在一个字典中，通过实体的mention名字找到所有可能的id
class MentionID:
    def __init__(self):
        self.id_set = {}
        self.mention = {}

    # def load_mention_2_id(self, knowledge_base_mention_file_name=gl.knowledge_base_mention_file_name):
    def load_mention_2_id(self, knowledge_base_mention_file_name):
        #knowledge_base_mention_file_name = '../data/nlpcc-iccpol-2016.kbqa.kb.mention2id'
        # Typically, it takes ~60s to load this file.
        t1 = time.time()
        fh = codecs.open(knowledge_base_mention_file_name, 'r', encoding='utf-8')
        for line_no, line in enumerate(fh.readlines()):
            try:
                mention, aliases = line.rstrip().split(' ||| ')
            except ValueError:
                # print 'Error at line', line_no
                # print 'Line content:', line.rstrip()
                continue
            aliases_list = aliases.split('\t')
            self.id_set[mention] = set(aliases_list)
            # for alias in aliases_list:
            #     self.mention[alias] = mention
        fh.close()
        t2 = time.time()
        print('Loading mention2id consumed', t2 - t1, 'seconds')

    def show(self, first_n=20):
        for i, mention in enumerate(self.id_set.keys()):
            print(i, mention)
            for alias in self.id_set[mention]:
                print('Possible alias:', alias)
            if i > first_n:
                break

    def show_id_set(self, mention):
        if mention in self.id_set:
            print(mention, 'in mention2id')
            for some_id in self.id_set[mention]:
                print(some_id)
            return self.id_set[mention]
        else:
            print('Unfortunately,' + mention + 'not found!')
            return set([])

    def find_id_set(self, mention):
        if mention in self.id_set:
            return self.id_set[mention]
        else:
            return set([])


#def load_qa_data(path, encode='utf-8'):
 #   with open(path, 'r', encoding=encode) as fi:
