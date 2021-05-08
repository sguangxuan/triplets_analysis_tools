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

    def get_keys(self, d, value):
        return [k for k, v in d.items() if v == value]

    def resavejson(self, path='dic.json'):

        f = open(path, 'r')
        content = f.read()
        unionTree = json.loads(content)
        f.close()

        queue = [unionTree]
        while True:
            for q in queue:
                queue.pop(queue.index(q))
                try:
                    del q['entities']
                except:
                    pass
                try:
                    del q['isleaf']
                except:
                    pass
                try:
                    del q['isroot']
                except:
                    pass
                if 'children' in q.keys():
                    keys = list(q['children'].keys())
                    for _ in keys:
                        if q['children'][_]['isleaf']:
                            del q['children'][_]
                    for _ in q['children'].keys():
                        queue.append(q['children'][_])
            print(len(queue))
            if not queue:
                break

        print(unionTree)

        item = json.dumps(unionTree)
        try:
            if not os.path.exists('dic-new.json'):
                with open('dic-new.json', "w", encoding='utf-8') as f:
                    f.write(item + "\n")
                    print("^_^ write success")
            else:
                with open('dic-new.json', "a", encoding='utf-8') as f:
                    f.write(item + "\n")
                    print("^_^ write success")
        except Exception as e:
            print("write error==>", e)


        self.analysis_entities()
        print(self.entities)
        item = json.dumps(self.entities)
        try:
            if not os.path.exists('dic-e.json'):
                with open('dic-e.json', "w", encoding='utf-8') as f:
                    f.write(item + "\n")
                    print("^_^ write success")
            else:
                with open('dic-e.json', "a", encoding='utf-8') as f:
                    f.write(item + "\n")
                    print("^_^ write success")
        except Exception as e:
            print("write error==>", e)


        return 0

    def classify_entities(self):
        """分析实体类别"""
        self.analysis_entities()
        # 尝试对entities归类（求并查集(union-find)问题）
        type_id = [0]
        classified = []
        union = {}
        unionTree = {}  # 这个Tree中的定义，根节点为unionTree，unionTree的子节点是并查集，各个子节点维护一个evidence(证据)的数值，叶子节点是小并查集

        for e in self.entities.keys():
            print(e)
            # 第一个实体
            if len(classified) == 0:
                self.entities[e]['type'] = max(type_id) + 1
                e_ = {'entities': [e],
                      'find': [self.entities[e]['relation_as_head'], self.entities[e]['relation_as_tail']],
                      'isleaf': True,
                      'isroot': False}
                unionTree = {'children': {max(type_id) + 2: e_},  # dic类型，注意可能是节点与叶子混合的
                             'type': max(type_id) + 1,
                             'evidence': 0,
                             'entities': [e],
                             'find': [self.entities[e]['relation_as_head'], self.entities[e]['relation_as_tail']],
                             'isleaf': False,
                             'isroot': True}

                type_id.append(max(type_id) + 1)
                type_id.append(max(type_id) + 1)
                classified.append(e)
                continue

            # 第一个之外的实体
            flag = False
            # 广度优先，先验证每一层找到该节点位置，当这一层有多个节点可合并的时候，说明该节点归类到这一层
            path = []    # 记录当前访问路径
            queue = [unionTree]   # 遍历队列
            while True:
                queue_ = queue
                queue = []
                path_ = path
                for q in queue_:

                    if not q['isleaf']:
                        node = q['children']
                    else:
                        node = q

                    if 'isleaf' in node.keys() and node['isleaf']:
                        if list(set(self.entities[e]['relation_as_head']) & set(node['find'][0])) or list(set(self.entities[e]['relation_as_tail']) & set(node['find'][1])):
                            queue.append(node)
                    else:
                        for k in node.keys():
                            # if type(node[k]) is list:
                            if list(set(self.entities[e]['relation_as_head']) & set(node[k]['find'][0])) or list(set(self.entities[e]['relation_as_tail']) & set(node[k]['find'][1])):
                                queue.append(node[k])


                    if len(queue) == 1:

                        if queue[0]['isleaf'] and q['isroot']:
                            # 只有一个，如果这个是叶子，父辈是root，则生成新节点
                            e_ = {'entities': [e],
                                  'find': [self.entities[e]['relation_as_head'],
                                           self.entities[e]['relation_as_tail']],
                                  'isleaf': True,
                                  'isroot': False}
                            children_ = {}
                            for child in queue:
                                if 'children' in q.keys():
                                    children_[self.get_keys(q['children'], child)[0]] = child


                            children_[max(type_id) + 1] = e_
                            type_id.append(max(type_id) + 1)

                            new_node = {'children': children_,
                                        'type': max(type_id) + 1,
                                        'evidence': 1,
                                        'entities': list(set([e] + [e_ for q_ in queue for e_ in q_['entities']])),
                                        'find': [list(set(
                                            self.entities[e]['relation_as_head'] + [e_ for q_ in queue for e_ in
                                                                                    q_['find'][0]])),
                                                 list(set(
                                                     self.entities[e]['relation_as_tail'] + [e_ for q_ in queue for
                                                                                             e_ in
                                                                                             q_['find'][1]]))],
                                        'isleaf': False,
                                        'isroot': False}
                            type_id.append(max(type_id) + 1)

                            for child in queue:
                                if 'children' in q.keys():
                                    del q['children'][self.get_keys(q['children'], child)[0]]

                            if 'children' in q.keys():
                                q['children'][new_node['type']] = new_node
                            queue = []

                        elif queue[0]['isleaf'] and (not q['isroot']):
                            # 只有一个，如果这个是叶子，而且父辈不是root，那作为兄弟节点加入
                            e_ = {'entities': [e],
                                  'find': [self.entities[e]['relation_as_head'], self.entities[e]['relation_as_tail']],
                                  'isleaf': True,
                                  'isroot': False}
                            q['children'][max(type_id) + 1] = e_
                            q['entities'] = list(set(q['entities'] + [e]))
                            q['find'] = [list(set(q['find'][0] + self.entities[e]['relation_as_head'])),
                                         list(set(q['find'][1] + self.entities[e]['relation_as_tail']))]
                            type_id.append(max(type_id) + 1)
                            classified.append(e)
                            queue = []

                        else:
                            # 只有一个，如果这个是节点，那检查孩子节点
                            continue


                    elif len(queue) == 0:   # 这种情况应该只在第一层出现
                        # 0个，说明节点与r[k]为兄弟
                        # TODO: 在进入时候的节点里添加这个信息
                        e_ = {'entities': [e],
                              'find': [self.entities[e]['relation_as_head'], self.entities[e]['relation_as_tail']],
                              'isleaf': True,
                              'isroot': False}

                        q['children'][max(type_id) + 1] = e_
                        q['entities'] = list(set(q['entities'] + [e]))
                        q['find'] = [list(set(q['find'][0] + self.entities[e]['relation_as_head'])),
                                     list(set(q['find'][1] + self.entities[e]['relation_as_tail']))]
                        type_id.append(max(type_id) + 1)
                        classified.append(e)
                        queue = []

                    elif len(queue) > 1:  # 要分辨有没有根节点，即是不是在第一层？
                        # 看看queue是不是纯叶子list
                        leaf_flag = True
                        for i_ in queue:
                            if not i_['isleaf']:
                                leaf_flag = False

                        if q['isroot']:
                            # 多个，应当在这一层合并
                            # TODO: 新建一个节点new_node，把这一层和这个e_合并起来，在进入时候的节点里添加这个new_node信息
                            e_ = {'entities': [e],
                                  'find': [self.entities[e]['relation_as_head'], self.entities[e]['relation_as_tail']],
                                  'isleaf': True,
                                  'isroot': False}
                            children_ = {}
                            for child in queue:
                                children_[self.get_keys(q['children'], child)[0]] = child
                            children_[max(type_id) + 1] = e_
                            type_id.append(max(type_id) + 1)

                            new_node = {'children': children_,
                                        'type': max(type_id) + 1,
                                        'evidence': 1,
                                        'entities': list(set([e] + [e_ for q_ in queue for e_ in q_['entities']])),
                                        'find': [list(set(self.entities[e]['relation_as_head'] + [e_ for q_ in queue for e_ in q_['find'][0]])),
                                                 list(set(self.entities[e]['relation_as_tail'] + [e_ for q_ in queue for e_ in q_['find'][1]]))],
                                        'isleaf': False,
                                        'isroot': False}
                            type_id.append(max(type_id) + 1)

                            for child in queue:
                                del q['children'][self.get_keys(q['children'], child)[0]]

                            q['children'][new_node['type']] = new_node
                            queue = []

                        elif leaf_flag:
                            # queue中纯叶子，且不是root
                            e_ = {'entities': [e],
                                  'find': [self.entities[e]['relation_as_head'], self.entities[e]['relation_as_tail']],
                                  'isleaf': True,
                                  'isroot': False}

                            q['children'][max(type_id) + 1] = e_
                            q['entities'] = list(set(q['entities'] + [e]))
                            q['find'] = [list(set(q['find'][0] + self.entities[e]['relation_as_head'])),
                                         list(set(q['find'][1] + self.entities[e]['relation_as_tail']))]
                            type_id.append(max(type_id) + 1)
                            classified.append(e)
                            queue = []

                        else:
                            # node['evidence'] += 1
                            # 检查孩子节点
                            # 把queue中的叶子节点剔除
                            for _ in queue:
                                if _['isleaf']:
                                    queue.pop(queue.index(_))
                            continue

                if not queue:
                    break

        item = json.dumps(unionTree)
        try:
            if not os.path.exists('dic.json'):
                with open('dic.json', "w", encoding='utf-8') as f:
                    f.write(item + "\n")
                    print("^_^ write success")
            else:
                with open('dic.json', "a", encoding='utf-8') as f:
                    f.write(item + "\n")
                    print("^_^ write success")
        except Exception as e:
            print("write error==>", e)

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

    def neo4j_to_triplets(self, addr, delimiter='\t', use_id=True):
        if use_id:
            grapha = self.graph.run("MATCH (a)-[r]->(b) RETURN id(a), type(r), id(b)").to_table()
            id2entity = self.graph.run("MATCH (a) RETURN id(a), a.name").to_table()
        else:
            grapha = self.graph.run("MATCH (a)-[r]->(b) RETURN a.name, type(r), b.name").to_table()
        with open(addr, 'w', newline="", encoding='utf-8') as f:
            grapha.write_separated_values(delimiter, file=f, header=None, skip=None, limit=None, newline='\r\n', quote='"')
        if use_id:
            with open(addr+"id2E.csv", 'w', newline="", encoding='utf-8') as f:
                id2entity.write_separated_values(delimiter, file=f, header=None, skip=None, limit=None, newline='\r\n', quote='"')


if __name__ == "__main__":
    addr = ['data/train.txt', 'data/valid.txt', 'data/test.txt']
    # addr = ['data/self-test.txt']
    # addr = ['data/triplets.csv']
    # 实例化
    triplets = triplets_analysis(addr)
    # 三元组统计
    triplets.save_result_to_CSV(savepath='result/')
    # 数据集分割
    # triplets.split_dataset(ratio='10:1:1', savepath='result/', conditional_random=True)

    # triplets.classify_entities()
    # 简化保存的json格式，并重新保存
    # triplets.resavejson(path='dicCMKG.json')
    triplets.resavejson(path='dicYAGO10.json')

    # 三元组导入Neo4j
    # triplets.triplets_to_Neo4j("http://127.0.0.1//:7474", 'testGraph', '123456', deleteAll=True)

    # 实例化
    # neo4j = Neo4j_analysis("http://127.0.0.1//:7474", 'testGraph', '123456')
    # Neo4j导出三元组
    # neo4j.neo4j_to_triplets('result/triplets.csv')



