from collections import OrderedDict, defaultdict
import json
import pickle

"""
1.读取all_50_schemas,把关系做编码
2.把text都取出来，用于产生字典编码
3.把spo_list取出来，重组
4.生成每个样本
"""

# 关系
with open('./data_deal/p.pkl', 'rb') as f:
    p_index, index_p = pickle.load(f)


# 制作pos标签
def postag2list(postag):
    postag_list = []
    for i in postag:
        word = i['word']
        pos = i['pos']
        postag_list += (len(word) * [pos])

    return postag_list


test2_data_process = []
with open('./data/test2_data_postag.json', 'r', encoding='utf-8') as f:
    for idx, line_raw in enumerate(f):
        line_raw = json.loads(line_raw)
        text = line_raw['text']

        text_pos = line_raw['postag']
        postag_list = postag2list(text_pos)
        if len(postag_list) != len(text):
            print(idx)
            postag_list = ['n'] * len(text)

        test2_data_process.append({'text': line_raw['text'],
                                   'text_pos': postag_list})
        if (idx + 1) % 10000 == 0:
            print('finish %d' % (idx + 1))

# 词典
with open('./data_deal/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

for num_words in [6000, 7000]:
    with open('./data_deal/%d/word_index.pkl' % num_words, 'rb') as f:
        word_index = pickle.load(f)

    # pos词典
    with open('./data_deal/tokenizer_pos.pkl', 'rb') as f:
        tokenizer_pos = pickle.load(f)
    with open('./data_deal/pos_index.pkl', 'rb') as f:
        pos_index = pickle.load(f)

    # 开始处理原始数据
    for i in test2_data_process:
        i['text_seq'] = [word_index.get(c, num_words + 1) for c in i['text']]
        i['pos_seq'] = [pos_index.get(c, len(pos_index) + 1) for c in i['text_pos']]
    with open('./data_deal/%d/test2_data_process.pkl' % num_words, 'wb') as f:
        pickle.dump(test2_data_process, f)
