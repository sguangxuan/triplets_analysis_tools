'''
用于数据集的分析
输入csv文件列表
统计三元组数量、头实体数量、尾实体数量、实体数量、关系种类、关系个数
对三元组实体进行分类
输出统计结果到csv保存
'''
import csv
import itertools
import copy
import json
import math
import os
import random


def dedupe(items, key=None):
    seen = set()
    for item in items:
        # - 不可哈希值转可哈希值部分(核心)
        val = item if key is None else key(item)
        if val not in seen:
            yield item# 生成器 更省心
            seen.add(val)

class triplets_analysis:
    def __init__(self, addr, delimiter='\t'):
        self.triplet_addr = addr
        self.delimiter = delimiter

        # 读取三元组
        self.triplets_file = {}
        self.openfile()
        self.triplets_file_analysis = {}
        self.triplets_all_analysis = {}
        self.entities = {}
        self.relations = {}
        self.analysis()

        self.result = {}

    def analysis(self):
        """统计每个数据集"""
        for file_addr in self.triplet_addr:
            self.triplets_file_analysis[file_addr] = {}
            self.triplets_file_analysis[file_addr]['head'] = []
            self.triplets_file_analysis[file_addr]['relation'] = []
            self.triplets_file_analysis[file_addr]['tail'] = []
            for t in self.triplets_file[file_addr]:
                self.triplets_file_analysis[file_addr]['head'].append(t[0])
                self.triplets_file_analysis[file_addr]['relation'].append(t[1])
                self.triplets_file_analysis[file_addr]['tail'].append(t[2])
            self.triplets_file_analysis[file_addr]['head'] = list(set(self.triplets_file_analysis[file_addr]['head']))
            self.triplets_file_analysis[file_addr]['relation'] = list(set(self.triplets_file_analysis[file_addr]['relation']))
            self.triplets_file_analysis[file_addr]['tail'] = list(set(self.triplets_file_analysis[file_addr]['tail']))
            self.triplets_file_analysis[file_addr]['number'] = {'head_number': len(self.triplets_file_analysis[file_addr]['head']),
                                                                'relation_type_number': len(self.triplets_file_analysis[file_addr]['relation']),
                                                                'tail_number': len(self.triplets_file_analysis[file_addr]['tail']),
                                                                'entity_number': len(list(dedupe(self.triplets_file_analysis[file_addr]['head']+self.triplets_file_analysis[file_addr]['tail'], key=lambda x: tuple(x)))),
                                                                'relation_number': len(list(dedupe(self.triplets_file[file_addr], key=lambda x: tuple(x))))}

        """统计总数据集"""
        self.triplets_all_analysis['head'] = []
        self.triplets_all_analysis['relation'] = []
        self.triplets_all_analysis['tail'] = []
        self.triplets_all_analysis['entity'] = []
        for file_addr in self.triplet_addr:
            self.triplets_all_analysis['head'].extend(self.triplets_file_analysis[file_addr]['head'])
            self.triplets_all_analysis['relation'].extend(self.triplets_file_analysis[file_addr]['relation'])
            self.triplets_all_analysis['tail'].extend(self.triplets_file_analysis[file_addr]['tail'])
        self.triplets_all_analysis['entity'] = self.triplets_all_analysis['head'] + self.triplets_all_analysis['tail']

        self.triplets_all_analysis['head'] = list(set(self.triplets_all_analysis['head']))
        self.triplets_all_analysis['relation'] = list(set(self.triplets_all_analysis['relation']))
        self.triplets_all_analysis['tail'] = list(set(self.triplets_all_analysis['tail']))
        self.triplets_all_analysis['entity'] = list(set(self.triplets_all_analysis['entity']))
        triplets_tmp = []
        for file_addr in self.triplet_addr:
            triplets_tmp.extend(self.triplets_file[file_addr])
        self.triplets_all_analysis['number'] = {'head_number': len(self.triplets_all_analysis['head']),
                                                'relation_type_number': len(self.triplets_all_analysis['relation']),
                                                'tail_number': len(self.triplets_all_analysis['tail']),
                                                'entity_number': len(self.triplets_all_analysis['entity']),
                                                'relation_number': len(list(dedupe(triplets_tmp, key=lambda x: tuple(x))))}


        '''print([self.triplets_file_analysis[file_addr]['number'] for file_addr in self.triplet_addr])
        print(self.triplets_all_analysis['number'])'''
        return self.triplets_file_analysis, self.triplets_all_analysis

    def triplets_to_Neo4j(self, ip, username, password, deleteAll=False):
        from py2neo import Node, Graph, Relationship, NodeMatcher
        print('对实体分类...')
        union = self.classify_entities()

        print('分类完成，构建图谱...')
        """建立连接"""
        self.graph = Graph(ip, username=username, password=password)
        if deleteAll:
            self.graph.delete_all()
        """建立节点"""
        for node in self.triplets_all_analysis['entity']:
            print(self.entities[node]['type'], node)
            name_node = Node(str(self.entities[node]['type']), name=str(node))
            self.graph.create(name_node)
        """建立关系"""
        matcher = NodeMatcher(self.graph)

        m = 0
        for key in self.triplets_file.keys():
            for t in self.triplets_file[key]:
                try:
                    rel = Relationship(matcher.match(str(self.entities[str(t[0])]['type']), name=str(t[0])).first(),
                                       str(t[1]),
                                       matcher.match(str(self.entities[str(t[2])]['type']), name=str(t[2])).first())
                    self.graph.create(rel)
                except:
                    print('Error:')
                m += 1
                print(m, t)

    def analysis_entities(self):
        """统计全部实体情况"""
        for e in self.triplets_all_analysis['entity']:
            self.entities[e] = {'type': 0,
                                'relation_as_head': [],
                                'relation_as_tail': [],
                                'count': 0}
        for e in self.triplets_all_analysis['relation']:
            self.relations[e] = {'head': [],
                                 'tail': []}
        for file_addr in self.triplet_addr:
            for t in self.triplets_file[file_addr]:
                head = t[0]
                relation = t[1]
                tail = t[2]
                self.relations[relation]['head'] = head
                self.relations[relation]['tail'] = tail
                if relation not in self.entities[head]['relation_as_head']:
                    self.entities[head]['relation_as_head'].append(relation)
                    self.entities[head]['count'] += 1
                if relation not in self.entities[tail]['relation_as_tail']:
                    self.entities[tail]['relation_as_tail'].append(relation)
                    self.entities[tail]['count'] += 1

    def classify_entities(self):
        """分析实体类别"""
        self.analysis_entities()
        # 尝试对entities归类（求并查集(union-find)问题）
        type_id = [0]
        classified = []
        union = {}
        for e in self.entities.keys():
            if len(classified) == 0:
                self.entities[e]['type'] = max(type_id) + 1
                union[str(max(type_id) + 1)] = {'entities': [e], 'find': [self.entities[e]['relation_as_head'], self.entities[e]['relation_as_tail']]}
                type_id.append(max(type_id) + 1)
                classified.append(e)
                continue
            flag = False
            for h in union.keys():
                if list(set(self.entities[e]['relation_as_head']) & set(union[h]['find'][0])) or list(set(self.entities[e]['relation_as_tail']) & set(union[h]['find'][1])):
                    # 有交集
                    self.entities[e]['type'] = h
                    classified.append(e)
                    union[h]['entities'].append(e)
                    union[h]['find'][0].extend(self.entities[e]['relation_as_head'])
                    union[h]['find'][0] = list(set(union[h]['find'][0]))
                    union[h]['find'][1].extend(self.entities[e]['relation_as_tail'])
                    union[h]['find'][1] = list(set(union[h]['find'][1]))
                    flag = True
                    break
            if flag:
                tmp_union = copy.deepcopy(union)
                while True:
                    for h in itertools.combinations(union.keys(), 2):
                        try:
                            if list(set(union[str(h[0])]['find'][0]) & set(union[str(h[1])]['find'][0])) or list(set(union[str(h[0])]['find'][1]) & set(union[str(h[1])]['find'][1])):
                                # 合并
                                for tmp in union[str(h[1])]['entities']:
                                    self.entities[tmp]['type'] = h[0]
                                union[str(h[0])]['entities'].extend(union[str(h[1])]['entities'])
                                union[str(h[0])]['find'][0].extend(union[str(h[1])]['find'][0])
                                union[str(h[0])]['find'][0] = list(set(union[str(h[0])]['find'][0]))
                                union[str(h[0])]['find'][1].extend(union[str(h[1])]['find'][1])
                                union[str(h[0])]['find'][1] = list(set(union[str(h[0])]['find'][1]))
                                del union[str(h[1])]
                        except:
                            pass
                    if tmp_union == union:
                        break
                    else:
                        tmp_union = union

            if not flag:
                self.entities[e]['type'] = max(type_id) + 1
                union[str(max(type_id) + 1)] = {'entities': [e], 'find': [self.entities[e]['relation_as_head'], self.entities[e]['relation_as_tail']]}
                type_id.append(max(type_id) + 1)
                classified.append(e)
        '''for e in self.entities.keys():
            print(e, self.entities[e]['type'], self.entities[e]['relation_as_head'], self.entities[e]['relation_as_tail'])
        print(union)'''

        '''item = json.dumps(union)
        try:
            if not os.path.exists('dic.json'):
                with open('dic.json', "w", encoding='utf-8') as f:
                    f.write(item + ",\n")
                    print("^_^ write success")
            else:
                with open('dic.json', "a", encoding='utf-8') as f:
                    f.write(item + ",\n")
                    print("^_^ write success")
        except Exception as e:
            print("write error==>", e)'''

        return union

    def split_dataset(self, ratio='8:1:1', savepath='./', conditional_random=True):
        """
        分割数据集
        ratio为比例，string格式，两个英文“:”分隔，依次为训练集、验证集、测试集。
        "0"或""表示不生成该集合，"-"表示其他值为数据集条数，"-"所在的集和平分余下的数据
        conditional_random为True时，生成的训练集中含有全部实体及关系，保证所训练嵌入空间包含全部节点和边。
        """
        split = ratio.split(':')
        train_r = (split[0] if split[0] != '' else 0)
        valid_r = (split[1] if split[1] != '' else 0)
        test_r = (split[2] if split[2] != '' else 0)
        # 存在"-"，其他值为数量，"-"所在的集和平分余下的数据
        if '-' in split:
            sum = 0
            count = 0
            index = []
            for i, e in enumerate(split):
                if e == '-':
                    count += 1
                    index.append(i)
                else:
                    sum = sum + eval(e)
            trian = eval(train_r) if train_r != '-' else (self.triplets_all_analysis['number']['relation_number'] - min(sum, self.triplets_all_analysis['number']['relation_number'])) / count
            valid = eval(valid_r) if valid_r != '-' else (self.triplets_all_analysis['number']['relation_number'] - min(sum, self.triplets_all_analysis['number']['relation_number'])) / count
            test = eval(test_r) if test_r != '-' else (self.triplets_all_analysis['number']['relation_number'] - min(sum, self.triplets_all_analysis['number']['relation_number'])) / count
            # 取整
            if count > 1:
                for i in range(count-1):
                    if index[i] == 0:
                        trian = math.floor(trian)
                    elif index[i] == 1:
                        valid = math.floor(valid)
                    elif index[i] == 2:
                        test = math.floor(test)
                if index[-1] == 0:
                    trian = self.triplets_all_analysis['number']['relation_number'] - valid - test
                elif index[-1] == 1:
                    valid = self.triplets_all_analysis['number']['relation_number'] - trian - test
                elif index[-1] == 2:
                    test = self.triplets_all_analysis['number']['relation_number'] - valid - trian
        # 不存在"-"，考虑比例
        else:
            sum = eval(train_r) + eval(valid_r) + eval(test_r)
            valid = math.floor(self.triplets_all_analysis['number']['relation_number'] * eval(valid_r) / sum)
            test = math.floor(self.triplets_all_analysis['number']['relation_number'] * eval(test_r) / sum)
            trian = self.triplets_all_analysis['number']['relation_number'] - valid - test

        # 分割数据集
        dataset = []
        train_dataset = []
        valid_dataset = []
        test_dataset = []
        for file_addr in self.triplet_addr:
            for t in self.triplets_file[file_addr]:
                dataset.append(t)
        # 随机分割
        if not conditional_random:
            candidateindex = list(range(0, len(dataset)))
            randomindex = random.sample(candidateindex, len(dataset))
            train_dataset = [dataset[i] for i in randomindex[0: trian]]
            valid_dataset = [dataset[i] for i in randomindex[trian: trian + valid]]
            test_dataset = [dataset[i] for i in randomindex[trian + valid: trian + valid + test]]
        # 考虑嵌入
        else:
            # candidataset候选集, singledataset不可放进train的候选集, selectdataseta选过的集和, count存储各个节点或关系的数量
            candidataset = dataset
            singledataset = []
            selectdataset = []
            count = {'entity': {}, 'relation': {}}
            for i in self.triplets_all_analysis['entity']:
                count['entity'][i] = 0
            for i in self.triplets_all_analysis['relation']:
                count['relation'][i] = 0
            # 统计各个实体、关系数量
            for t in candidataset:
                count['entity'][t[0]] += 1
                count['entity'][t[2]] += 1
                count['relation'][t[1]] += 1
            entity = list(count['entity'].keys())
            relation = list(count['relation'].keys())
            i = 0
            while True:
                i += 1
                if i % 100 == 0:
                    print(i)
                if entity:
                    tmp = []
                    for t in candidataset:
                        if t[0] == entity[0] or t[2] == entity[0]:
                            tmp.append(t)
                    tmp_t = tmp[random.randint(0, len(tmp)-1)]
                    selectdataset.append(candidataset.pop(candidataset.index(tmp_t)))
                    if tmp_t[0] in entity:
                        entity.pop(entity.index(tmp_t[0]))
                    if tmp_t[1] in relation:
                        relation.pop(relation.index(tmp_t[1]))
                    if tmp_t[2] in entity:
                        entity.pop(entity.index(tmp_t[2]))
                elif relation:
                    tmp = []
                    for t in candidataset:
                        if t[1] == relation[0]:
                            tmp.append(t)
                    tmp_t = tmp[random.randint(0, len(tmp) - 1)]
                    selectdataset.append(candidataset.pop(candidataset.index(tmp_t)))
                    if tmp_t[0] in entity:
                        entity.pop(entity.index(tmp_t[0]))
                    if tmp_t[1] in relation:
                        relation.pop(relation.index(tmp_t[1]))
                    if tmp_t[2] in entity:
                        entity.pop(entity.index(tmp_t[2]))
                else:
                    break
            if len(selectdataset) > trian:
                return 0
            randomindex = random.sample(range(0, len(candidataset)), len(candidataset))
            train_dataset = selectdataset + [candidataset[i] for i in randomindex[0: trian-len(selectdataset)]]
            valid_dataset = [candidataset[i] for i in randomindex[trian-len(selectdataset): trian - len(selectdataset)+valid]]
            test_dataset = [candidataset[i] for i in randomindex[trian - len(selectdataset)+valid: trian - len(selectdataset)+valid+test]]

        if trian != 0:
            self.save_to_CSV(train_dataset, savepath+'train_new.csv', delimiter='\t')
        if valid != 0:
            self.save_to_CSV(valid_dataset, savepath+'valid_new.csv', delimiter='\t')
        if test != 0:
            self.save_to_CSV(test_dataset, savepath+'test_new.csv', delimiter='\t')

        print(train_dataset, valid_dataset, test_dataset)
        print(self.triplets_all_analysis['number']['relation_number'])
        print(trian, valid, test)

        return 1

    def openfile(self):
        for file_addr in self.triplet_addr:
            self.triplets_file[file_addr] = []
            with open(file_addr, encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=self.delimiter)
                for row in reader:
                    if len(row) == 3:
                        self.triplets_file[file_addr].append(row)
                    else:
                        pass

    def save_result_to_CSV(self, savepath='./'):
        with open(savepath+'analysis_result.csv', 'w', newline="", encoding='utf-8') as f:
            csv_write = csv.writer(f)
            csv_head = ['name', 'head_number', 'relation_type_number', 'tail_number', 'entity_number', 'triplet_number']
            csv_write.writerow(csv_head)
            for file_addr in self.triplet_addr:
                csv_write.writerow([file_addr,
                                    self.triplets_file_analysis[file_addr]['number']['head_number'],
                                    self.triplets_file_analysis[file_addr]['number']['relation_type_number'],
                                    self.triplets_file_analysis[file_addr]['number']['tail_number'],
                                    self.triplets_file_analysis[file_addr]['number']['entity_number'],
                                    self.triplets_file_analysis[file_addr]['number']['relation_number']])
            csv_write.writerow(['sum',
                                self.triplets_all_analysis['number']['head_number'],
                                self.triplets_all_analysis['number']['relation_type_number'],
                                self.triplets_all_analysis['number']['tail_number'],
                                self.triplets_all_analysis['number']['entity_number'],
                                self.triplets_all_analysis['number']['relation_number']])

    def save_to_CSV(self, list, savepath, delimiter=','):
        with open(savepath, 'w', newline="", encoding='utf-8') as f:
            csv_write = csv.writer(f, delimiter=delimiter)
            for l in list:
                csv_write.writerow(l)

class Neo4j_analysis:
    def __init__(self, ip, username, password):
        from py2neo import Graph, cypher
        self.ip = ip
        self.username = username
        self.password = password
        self.graph = Graph(self.ip, username=self.username, password=self.password)

    def neo4j_to_triplets(self, addr, delimiter='\t'):
        grapha = self.graph.run("MATCH (a)-[r]-(b) RETURN a.name, type(r), b.name").to_table()
        with open(addr, 'w', newline="", encoding='utf-8') as f:
            grapha.write_separated_values(delimiter, file=f, header=None, skip=None, limit=None, newline='\r\n', quote='"')


if __name__ == "__main__":
    addr = ['data/train.txt', 'data/valid.txt', 'data/test.txt']
    # 实例化
    triplets = triplets_analysis(addr)
    # 三元组统计
    # triplets.save_result_to_CSV(savepath='result/')
    # 数据集分割
    # triplets.split_dataset(ratio='10:1:1', savepath='result/', conditional_random=True)
    # 三元组导入Neo4j
    triplets.triplets_to_Neo4j("http://127.0.0.1//:7474", 'testGraph', '123456', deleteAll=True)

    '''# 实例化
    neo4j = Neo4j_analysis("http://127.0.0.1//:7474", 'testGraph', '123456')
    # Neo4j导出三元组
    neo4j.neo4j_to_triplets('result/triplets.csv')'''



