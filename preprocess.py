from collections import OrderedDict, defaultdict
import json
import pickle
from keras.preprocessing.text import Tokenizer
import os

"""
1.读取all_50_schemas,把关系做编码
2.把text都取出来，用于产生字典编码
3.把spo_list取出来，匹配首尾，重组为{
    s1:[[p11,o11],[p12,o12],...],
    s2:[[p21,o21],[p22,o22],...],
    ...
}
4.增加crf序列标注（效果不理想）
{
    PAD_TAG: 0,
    START_TAG: 1,
    STOP_TAG: 2,
    'U':3
    'B': 4, 'M': 5, 'E': 6,
    'S': 7,
}
5.生成每个样本{
    text:...,
    text_seq:{[...],
    s_start_label:[1,0,0,0,1,0,...],
    s_end_label:[0,1,0,0,0,1,...],
    s_crf:[4,4,4,5,6,7,4,4,8,...],
    spo_list:[
        [[s1_start_index,s1_end_index],[8,0,4,0,0,...],[0,8,0,4,0,...]],
        [[s2_start_index,s2_end_index],[0,0,0,5,0,...],[0,0,0,0,5,...]],
        ...
    }
}
"""

num_words = 7000

# 关系预处理
if os.path.exists('./data_deal/p.pkl'):
    with open('./data_deal/p.pkl', 'rb') as f:
        p_index, index_p = pickle.load(f)
else:
    with open('./data/all_50_schemas', 'r', encoding='utf-8') as f:
        p_set = []
        for line_raw in f:
            line_raw = json.loads(line_raw)
            p_set.append(line_raw["predicate"])
        p_set = sorted(list(set(p_set)))  # 避免set无序不一致
        p_index = {p: (idx + 1) for idx, p in enumerate(p_set)}
        index_p = {(idx + 1): p for idx, p in enumerate(p_set)}

    with open('./data_deal/p.pkl', 'wb') as f:
        pickle.dump([p_index, index_p], f)

# 实体预处理
texts = []
texts_pos = []

# 词性转序列
def postag2list(postag):
    postag_list = []
    for i in postag:
        word = i['word']
        pos = i['pos']
        postag_list += (len(word) * [pos])

    return postag_list


def process_data(file_path):
    data_process = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line_raw in enumerate(f):
            line_raw = json.loads(line_raw)

            # 跳过没有信息抽取的
            if line_raw['spo_list']:
                text = line_raw['text']
                texts.append(text)

                text_pos = line_raw['postag']
                postag_list = postag2list(text_pos)
                if len(postag_list) != len(text):
                    print('pos错误：',idx)
                    postag_list = ['n'] * len(text)
                texts_pos.append(postag_list)

                line_process = OrderedDict()
                line_process['text'] = text
                line_process['text_pos'] = postag_list
                spo_list_raw = []

                # 拼接三元组,先记录索引,后续再转编码
                spo_dict = defaultdict(list)
                for spo in line_raw['spo_list']:
                    # 定位subject起止位置
                    s = spo['subject']
                    s_start = text.find(s)

                    # 定位object起止位置
                    o = spo['object']
                    o_start = text.find(o)

                    if s_start != -1 and o_start != -1:
                        s_start_end = (s_start, s_start + len(s) - 1)
                        o_start_end = (o_start, o_start + len(o) - 1)
                        p = p_index[spo['predicate']]
                        spo_dict[s_start_end].append([p, o_start_end])
                        spo_list_raw.append((s, spo['predicate'], o))
                if spo_dict:
                    spo_list = []
                    s_start_label = [0] * len(text)
                    s_end_label = [0] * len(text)
                    s_crf_label = [3] * len(text)
                    for s, po_list in spo_dict.items():
                        # subject起止位置=1,其余=0
                        s_start_label[s[0]] = 1
                        s_end_label[s[1]] = 1

                        # subject长度=1
                        if s[1] - s[0] == 0:
                            s_crf_label[s[0]] = 7
                        # subject长度>1
                        else:
                            s_crf_label[s[0]] = 4
                            s_crf_label[s[1]] = 6
                            s_crf_label[(s[0] + 1):s[1]] = [5] * (s[1] - s[0] - 1)

                        # object起止位置=关系编码,其余=0
                        o_start_labels = [0] * len(text)
                        o_end_labels = [0] * len(text)
                        for po in po_list:
                            p = po[0]
                            o_start, o_end = po[1]
                            o_start_labels[o_start] = p
                            o_end_labels[o_end] = p
                        spo_list.append([s, o_start_labels, o_end_labels])

                    line_process['s_start_label'] = s_start_label
                    line_process['s_end_label'] = s_end_label
                    line_process['s_crf_label'] = s_crf_label
                    line_process['spo_list'] = spo_list
                    line_process['spo_list_raw'] = spo_list_raw
                    data_process.append(line_process)
            if (idx + 1) % 10000 == 0:
                print('finish %d' % (idx + 1))
    return data_process


train_data_process = process_data('./data/train_data.json')
dev_data_process = process_data('./data/dev_data.json')

test1_data_process = []
with open('./data/test1_data_postag.json', 'r', encoding='utf-8') as f:
    for idx, line_raw in enumerate(f):
        line_raw = json.loads(line_raw)
        text = line_raw['text']
        texts.append(text)

        text_pos = line_raw['postag']
        postag_list = postag2list(text_pos)
        if len(postag_list) != len(text):
            print(idx)
            postag_list = ['n'] * len(text)
        texts_pos.append(postag_list)

        test1_data_process.append({'text': line_raw['text'],
                                   'text_pos': postag_list})
        if (idx + 1) % 10000 == 0:
            print('finish %d' % (idx + 1))

# 构建词典
if os.path.exists('./data_deal/tokenizer.pkl'):
    with open('./data_deal/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
else:
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(texts)

if os.path.exists('./data_deal/%d/word_index.pkl' % num_words):
    with open('./data_deal/%d/word_index.pkl' % num_words, 'rb') as f:
        word_index = pickle.load(f)
else:
    total_num = len(tokenizer.word_index)
    word_index = {}
    for word in tokenizer.word_index:
        word_id = tokenizer.word_index[word]
        if total_num < num_words:
            word_index = tokenizer.word_index
            num_words = total_num
        else:
            if word_id <= num_words:
                word_index.update({word: word_id})

# 构建pos词典
if os.path.exists('./data_deal/tokenizer_pos.pkl'):
    with open('./data_deal/tokenizer_pos.pkl', 'rb') as f:
        tokenizer_pos = pickle.load(f)
else:
    tokenizer_pos = Tokenizer()
    tokenizer_pos.fit_on_texts(texts_pos)

if os.path.exists('./data_deal/pos_index.pkl'):
    with open('./data_deal/pos_index.pkl', 'rb') as f:
        pos_index = pickle.load(f)
else:
    pos_index = tokenizer_pos.word_index

# 开始处理原始数据
for i in train_data_process:
    i['text_seq'] = [word_index.get(c, num_words + 1) for c in i['text']]
    i['pos_seq'] = [pos_index.get(c, len(pos_index) + 1) for c in i['text_pos']]
with open('./data_deal/%d/train_data_process.pkl' % num_words, 'wb') as f:
    pickle.dump(train_data_process, f)

for i in dev_data_process:
    i['text_seq'] = [word_index.get(c, num_words + 1) for c in i['text']]
    i['pos_seq'] = [pos_index.get(c, len(pos_index) + 1) for c in i['text_pos']]
with open('./data_deal/%d/dev_data_process.pkl' % num_words, 'wb') as f:
    pickle.dump(dev_data_process, f)

for i in test1_data_process:
    i['text_seq'] = [word_index.get(c, num_words + 1) for c in i['text']]
    i['pos_seq'] = [pos_index.get(c, len(pos_index) + 1) for c in i['text_pos']]
with open('./data_deal/%d/test1_data_process.pkl' % num_words, 'wb') as f:
    pickle.dump(test1_data_process, f)

with open('./data_deal/%d/word_index.pkl' % num_words, 'wb') as f:
    pickle.dump(word_index, f)

with open('./data_deal/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

with open('./data_deal/pos_index.pkl', 'wb') as f:
    pickle.dump(pos_index, f)

with open('./data_deal/tokenizer_pos.pkl', 'wb') as f:
    pickle.dump(tokenizer_pos, f)
