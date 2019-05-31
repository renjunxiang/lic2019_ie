import codecs
import torch
import pickle
import os
import json
import codecs
import re
from sklearn.externals import joblib
import numpy as np

clf = joblib.load('./ensemble_model.m')

DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

EMBEDDING_DIM = 128
num_layers = 3
HIDDEN_DIM = 512
BATCH_SIZE = 64
num_words = 7000

with open(DIR + '/data_deal/p.pkl', 'rb') as f:
    p_index, index_p = pickle.load(f)
with open(DIR + '/data_deal/%d/word_index.pkl' % num_words, 'rb') as f:
    word_index = pickle.load(f)
with open(DIR + '/data_deal/%d/test1_data_process.pkl' % num_words, 'rb') as f:
    test1_data_process = pickle.load(f)

p_count_text = {i: 0 for i in p_index}
p_count_spo = {i: 0 for i in p_index}

test1_predicts = []

file_name_old = {
    6000: {
        'baidubaike': {
            'match_4_lstm_768_4_1_2': [  # y
                6,  # 0.787
            ],
            'match_4_lstm_2048_4_1_2': [  # y
                6,  # 0.790
                7,  # 0.788
                5,  # 0.787
                10,  # 0.786
                9, 8,  # 0.785
            ],
            'match_5_lstm_1024_4_1_2': [  # 忘记
                5,  # 0.789
                6,  # 0.787
                8, 9,  # 0.785
            ],
        },
        'renmin': {
            'match_4_lstm_768_4_1_2': [
                6,  # 0.785
            ],
            'match_4_lstm_2048_4_1_2': [
                7,  # 0.786
                8,  # 0.785
            ],
            'match_5_lstm_1024_4_1_2': [  # 搞错了应该在7000里面
                # 7,  # 0.785
            ],
        },
        'sogounews': {
            'match_4_lstm_768_4_1_2': [
                5,  # 0.787
                6,  # 0.786
            ],
            'match_4_lstm_2048_4_1_2': [
                5, 7, 6,  # 0.785
            ],
        }
    },
    7000: {
        'baidubaike': {
            'match_4_lstm_768_4_1_2': [
                6,  # 0.786
                7,  # 0.785
            ],
            'match_4_lstm_2048_4_1_2': [
                5,  # 0.790
                6,  # 0.787
                7,  # 0.785
            ],
            'match_5_lstm_1024_4_1_2': [
                6,  # 0.787
                7,  # 0.786
                9, 8,  # 0.785
            ],
        },
        'renmin': {
            'match_4_lstm_2048_4_1_2': [
                7,  # 0.788
            ],
            'match_5_lstm_1024_4_1_2': [  # 6000那个弄错了
                7, 7,  # 0.785
            ],
        },
        'sogounews': {
            'match_4_lstm_2048_4_1_2': [
                7,  # 0.790
                6,  # 0.789
                8,  # 0.786
                9,  # 0.785
            ],
            'match_5_lstm_1024_4_1_2': [
                8,  # 0.786
                10, 6,  # 0.785
            ],
        }
    }
}

for num_words, value1 in file_name_old.items():
    for embedding_name, value2 in value1.items():
        for model_name, model_idxs in value2.items():
            for model_idx in model_idxs:
                with open('./models_old/%d/%s/%s/test1_%03d.pkl' % (num_words, embedding_name, model_name, model_idx),
                          'rb') as f:
                    test1 = pickle.load(f)
                    test1_predicts.append(test1)

file_name = {
    6000: {
        'baidubaike': {
            'match_4_lstm_768_50_4_1_2': [  # y
                6,  # 0.789
            ],
            'match_4_lstm_2048_50_4_1_2': [  # y
                6,  # 0.790
                4, 9,  # 0.788
            ],
            'match_4_lstm_2048_60_4_1_2': [  # y
                7, 10, 5,  # 0.790
                8,  # 0.789
            ],
            'match_4_lstm_2048_70_4_1_2': [  # 新
                6, 7, 5,  # 0.790
                9, 11, 8,  # 0.789
                10,  # 0.788
            ],
            'match_4_lstm_2048_80_4_1_2': [  # 新
                7, 6,  # 0.790
                9,  # 0.789
                4,  # 0.788
            ],
            'match_4_lstm_2048_100_4_1_2': [
                8, 6, 7, 9,  # 0.790
                10,  # 0.789
            ],
            'match_5_lstm_1024_50_4_1_2': [
                6,  # 0.790
                10, 7,  # 0.789
                8,  # 0.788
            ],
        },
        'renmin': {
            'match_4_lstm_2048_50_4_1_2': [  # x 错误
                8, 7,  # 0.789
            ],
            'match_4_lstm_2048_100_4_1_2': [
                6,  # 0.788
            ],
        },
        'sogounews': {
            'match_4_lstm_2048_50_4_1_2': [
                5,  # 0.790
            ],
            'match_4_lstm_2048_100_4_1_2': [
                8,  # 0.790
            ],
        }
    },
    7000: {
        'baidubaike': {
            'match_4_lstm_768_50_4_1_2': [  # y
                7,  # 0.789
            ],
            'match_4_lstm_768_100_4_1_2': [  # y
                8,  # 0.789
                9, 7,  # 0.788
            ],
            'match_4_lstm_2048_30_4_1_2': [  # y
                10,  # 0.790
                8, 7, 6,  # 0.788
            ],
            'match_4_lstm_2048_50_4_1_2': [  # y
                6, 5, 10, 8,  # 0.790
                9,  # 0.788
            ],
            'match_4_lstm_2048_60_4_1_2': [  # y
                8,  # 0.790
                9,  # 0.789
                10, 7, 6,  # 0.788
            ],
            'match_4_lstm_2048_70_4_1_2': [  # 新
                7, 8,  # 0.790
                5, 10,  # 0.789
                9, 12,  # 0.788
            ],
            'match_4_lstm_2048_100_4_1_2': [
                7,  # 0.790
                8, 6,  # 0.789
                9,  # 0.788
            ],
            'match_5_lstm_1024_50_4_1_2': [
                5,  # 0.790
                6, 8, 7,  # 0.788
            ],
            'match_5_lstm_1024_100_4_1_2': [
                6,  # 0.789
            ],
        },
        'renmin': {
            'match_5_lstm_1024_50_4_1_2': [
                7,  # 0.790 ?
                8,  # 0.788
            ],
            'match_4_lstm_2048_50_4_1_2': [
                7, 6,  # 0.788
            ],
            'match_4_lstm_2048_100_4_1_2': [
                7,  # 0.790
                8,  # 0.789
                11,  # 0.788
            ],
        },
        'sogounews': {
            'match_4_lstm_2048_50_4_1_2': [
                6,  # 0.790
                8,  # 0.788
            ],
            'match_4_lstm_2048_100_4_1_2': [
                7,  # 0.789
                10, 6,  # 0.788
            ],
        }
    }
}

for num_words, value1 in file_name.items():
    for embedding_name, value2 in value1.items():
        for model_name, model_idxs in value2.items():
            for model_idx in model_idxs:
                with open('./models/%d/%s/%s/test1_%03d.pkl' % (num_words, embedding_name, model_name, model_idx),
                          'rb') as f:
                    test1 = pickle.load(f)
                    test1_predicts.append(test1)

print('模型数量:', len(test1_predicts))


def find_most(x, rate=1.0):
    v = [[i, x.count(i)] for i in sorted(list(set(x)))]
    v = sorted(v, key=lambda x: x[1], reverse=True)
    # print(v[:4])

    result = None
    if len(v) > 1:
        if v[0][1] / v[1][1] >= rate:
            result = v[0][0]
    else:
        result = v[0][0]

    return result


def find_topk(x, k=1):
    v = [[i, x.count(i)] for i in sorted(list(set(x)))]
    v = sorted(v, key=lambda x: x[1], reverse=True)
    if len(v) < k:
        v += [[('None', 'None', 'None'), 0.00001] for _ in range(k - len(v))]

    return v[:k]


def submit(n):
    f = codecs.open('./result_0520_w_%d_sklearn.json' % n, 'w', encoding='utf-8')
    f1 = codecs.open('./research/result_0520_w_%d_sklearn.json' % n, 'w', encoding='utf-8')
    special1 = 0
    special2 = 0
    special3 = 0
    special4 = 0
    special5 = 0
    special6 = 0
    special7 = 0
    special8 = 0
    special9 = 0
    special10 = 0
    special11 = 0
    special12_1 = 0
    special12_2 = 0
    special13 = 0
    special14 = 0

    tpn = 0
    empty = []
    for i in range(len(test1)):
        text = test1_data_process[i]['text']
        for j in p_count_text:
            if j in text:
                p_count_text[j] += 1

        line_ensemble = []
        for test1_predict in test1_predicts:
            test1_predict_correct = [j for j in test1_predict[i] if
                                     not (j[0] == j[2] and (j[1] not in ['改编自', '所属专辑', '歌手', '作者']))]
            line_ensemble += test1_predict_correct

        line_ensemble_set = sorted(list(set(line_ensemble)))
        line_ensemble_new = [j for j in line_ensemble_set if line_ensemble.count(j) > n]

        # 开始规则部分

        # 规则0：一般情况下o不能包含/（已验证）
        line_ensemble_new_1 = []
        for j in line_ensemble_new:
            if ('/' in j[2]) and (j[1] in ['出版社', '作者', '作词', '作曲','身高','毕业院校']):
                s = j[0]
                p = j[1]
                o = j[2]
                for o1 in o.split('/'):
                    if len(o1) > 0:
                        line_ensemble_new_1.append((s, p, o1))
                        special13 += 1
            if ('/' in j[0]) and (j[1] in ['国籍']):
                s = j[0]
                p = j[1]
                o = j[2]
                for s1 in s.split('/'):
                    if len(s1) > 0:
                        line_ensemble_new_1.append((s1, p, o))
                        special13 += 1
            else:
                line_ensemble_new_1.append(j)

        line_ensemble_new = line_ensemble_new_1
        line_ensemble_new = sorted(list(set(line_ensemble_new)))

        # 规则1：一般情况下o不能包含顿号（已验证）
        line_ensemble_new_1 = []
        for j in line_ensemble_new:
            if '、' in j[2] and j[1] in ['出品公司', '作者', '出版社', '主演']:
                s = j[0]
                p = j[1]
                o = j[2]
                for o1 in o.split('、'):
                    line_ensemble_new_1.append((s, p, o1))
                    special1 += 1
            else:
                line_ensemble_new_1.append(j)

        line_ensemble_new = line_ensemble_new_1
        line_ensemble_new = sorted(list(set(line_ensemble_new)))

        # 规则2：xxx主演xxx客串，删客串（已验证）
        kechuan = re.findall('主演[\S]*客串[\S]*', text)
        if kechuan:
            line_ensemble_new_1 = []
            kechuan = kechuan[0]
            for j in line_ensemble_new:
                if not ((j[1] == '主演') and (j[2] in kechuan)):
                    line_ensemble_new_1.append(j)
                    special2 += 1

            line_ensemble_new = line_ensemble_new_1

        # 规则3：书名号没取全（已验证，0.002）
        shuming = re.findall('《[^《]+》', text)
        # shuming = ["《" + j[1:-1].strip() + "》" for j in shuming]
        shuming = [j[1:-1].strip() for j in shuming]
        if shuming:
            line_ensemble_new_1 = []
            for j in line_ensemble_new:
                s, p, o = j

                # s没有取全
                if s not in shuming:
                    for shuming_one in shuming:
                        if (s in shuming_one) and text.count(shuming_one) == text.count(s):
                            s = shuming_one
                            special3 += 1

                # o没有取全
                if (o not in shuming) and (p in ["所属专辑"]):
                    for shuming_one in shuming:
                        if (o in shuming_one) and text.count(shuming_one) == text.count(o):
                            o = shuming_one
                            special3 += 1

                # 关系错误
                if (s in shuming) and (p not in ["上映时间", "所属专辑", "导演", "出品公司", "制片人",
                                                 "编剧", "连载网站", "出版社", "主持人", "歌手",
                                                 "作词", "主角", "作者", "作曲", "嘉宾", "主演", "改编自"]):
                    special3 += 1
                    continue
                else:
                    line_ensemble_new_1.append((s, p, o))

            line_ensemble_new = line_ensemble_new_1
            line_ensemble_new = sorted(list(set(line_ensemble_new)))


        # 规则4：o错误来源于s、s错误来源o、so同一个（已验证）
        line_ensemble_new_1 = []
        for j in line_ensemble_new:
            s, p, o = j

            if ((o in s) or (s in o)) and (o != s) and (text.count(o) == text.count(s)):
                special4 += 1
                continue
            elif s == o and text.count(o) == 1:
                special4 += 1
                continue
            else:
                line_ensemble_new_1.append(j)

        line_ensemble_new = line_ensemble_new_1
        line_ensemble_new = sorted(list(set(line_ensemble_new)))

        # 规则5 占地面积（）
        mianji = re.findall('占地面积', text)
        if mianji:
            p_list = [j[1] for j in line_ensemble_new]
            if '占地面积' not in p_list:
                special5 += 1
                spo = find_topk(line_ensemble, 100)
                for j in spo:
                    if '占地面积' == j[0][1]:
                        line_ensemble_new.append(j[0])
                        break

        # 规则6 删不全的日期（已验证）
        p_list = [j[1] for j in line_ensemble_new]
        if len(set(["出生日期", "上映时间", "成立日期"]).intersection(set(p_list))) > 0:
            line_remove = []
            for j in line_ensemble_new:
                s, p, o = j
                o = re.sub('[\(\)]', '', o)
                if p in ["出生日期", "上映时间", "成立日期"]:
                    if re.findall(o + '[0-9年月]', text):
                        line_remove.append(j)
                        special6 += 1
                        riqi1 = re.findall(o + '[0-9]+月[0-9]+[日号]', text)
                        riqi2 = re.findall(o + '[0-9]+月[0-9]+', text)
                        riqi3 = re.findall(o + '[0-9]+月', text)
                        riqi4 = re.findall(o + '[0-9]+', text)
                        riqi5 = re.findall(o + '年', text)
                        riqi6 = re.findall(o + '[0-9]+[日号]', text)
                        riqi7 = re.findall(o + '月', text)

                        if riqi1:
                            line_ensemble_new.append((j[0], j[1], riqi1[0]))
                        elif riqi2:
                            line_ensemble_new.append((j[0], j[1], riqi2[0]))
                        elif riqi3:
                            line_ensemble_new.append((j[0], j[1], riqi3[0]))
                        elif riqi4:
                            line_ensemble_new.append((j[0], j[1], riqi4[0]))
                        elif riqi5:
                            line_ensemble_new.append((j[0], j[1], riqi5[0]))
                        elif riqi6:
                            line_ensemble_new.append((j[0], j[1], riqi6[0]))
                        elif riqi7:
                            line_ensemble_new.append((j[0], j[1], riqi7[0]))

            for j in line_remove:
                line_ensemble_new.remove(j)

        # 规则7 不全英文人名（已验证）
        renming = re.findall('[a-zA-Z][a-zA-Z\s]*[a-zA-Z]', text)
        if renming:
            line_ensemble_new_1 = []
            for j in line_ensemble_new:
                s, p, o = j

                # s没有取全
                if (s not in renming):
                    for renming_one in renming:
                        if (s in renming_one) and text.count(renming_one) == text.count(s):
                            s = renming_one
                            special7 += 1

                # o没有取全
                if (o not in renming):
                    for renming_one in renming:
                        if (o in renming_one) and text.count(renming_one) == text.count(o):
                            o = renming_one
                            special7 += 1
                line_ensemble_new_1.append((s, p, o))
            line_ensemble_new = line_ensemble_new_1
            line_ensemble_new = sorted(list(set(line_ensemble_new)))

        # 规则n：去掉首尾空格
        line_ensemble_new = [(j[0].strip(), j[1], j[2].strip()) for j in line_ensemble_new]

        # 规则8：少出版社三个字（已验证）
        line_ensemble_new_1 = []
        for j in line_ensemble_new:
            if (j[1] == '出版社') and ((j[2] + '出版社') in text):
                line_ensemble_new_1.append((j[0], j[1], j[2] + '出版社'))
                special8 += 1
            else:
                line_ensemble_new_1.append(j)
        line_ensemble_new = line_ensemble_new_1
        line_ensemble_new = sorted(list(set(line_ensemble_new)))

        # 规则9：公司名不全(修复后成功，0.0015)
        pattern = ['集团有限公司', '有限公司', '有限责任公司', '分公司', '集团', '公司', ]

        if re.findall('|'.join(pattern), text):
            line_ensemble_new_1 = []
            for j in line_ensemble_new:
                s, p, o = j
                for k in pattern:
                    if (s + k in text) and (text.count(s) == text.count(s + k)):
                        s = s + k
                        special9 += 1
                        break
                for k in pattern:
                    if (o + k in text) and (text.count(o) == text.count(o + k)):
                        o = o + k
                        special9 += 1
                        break
                line_ensemble_new_1.append((s, p, o))

            line_ensemble_new = line_ensemble_new_1
            line_ensemble_new = sorted(list(set(line_ensemble_new)))

        pattern = ['市', '省']

        if re.findall('|'.join(pattern), text):
            line_ensemble_new_1 = []
            for j in line_ensemble_new:
                s, p, o = j
                for k in pattern:
                    if (s + k in text) and (text.count(s) == text.count(s + k)):
                        s = s + k
                        # special9 += 1
                        break
                for k in pattern:
                    if (o + k in text) and (text.count(o) == text.count(o + k)):
                        o = o + k
                        # special9 += 1
                        break
                line_ensemble_new_1.append((s, p, o))
            line_ensemble_new = line_ensemble_new_1
            line_ensemble_new = sorted(list(set(line_ensemble_new)))

        # 规则10：剔除专辑被当成s（已验证）
        p_list = [j[1] for j in line_ensemble_new]
        if '所属专辑' in p_list:
            for j in line_ensemble_new:
                s, p, o = j
                if p == '所属专辑':
                    zhuanji = o
                    gequ = s
            line_ensemble_new_1 = []
            if zhuanji != gequ:
                for j in line_ensemble_new:
                    if j[0] == zhuanji:
                        special10 += 1
                    else:
                        line_ensemble_new_1.append(j)
                line_ensemble_new = line_ensemble_new_1
                line_ensemble_new = sorted(list(set(line_ensemble_new)))

        # 规则11：漏作曲（已验证）

        # 同时作词作曲
        zuoqu = re.search('(是|由|是由|，|、)+[^，、]+(作词|填词|写歌词)+[、]*(作曲|谱曲)', text)
        gequ = re.findall('《[^《]+》', text)
        p_list = [j[1] for j in line_ensemble_new]
        s_list = [j[0] for j in line_ensemble_new]
        if (len(gequ) == 1) and (gequ[0][1:-1] in s_list):
            for j in line_ensemble_new:
                # 避免书名号内不是歌曲
                if (j[0] == gequ[0][1:-1]) and (j[1] in ['导演', '出品公司', '制片人', '编剧', '主演', '改编自']):
                    gequ = gequ + gequ

        if zuoqu and ('作曲' not in p_list):
            zuoqu_add = []
            zuoqu = zuoqu[0]
            for j in line_ensemble_new:
                s, p, o = j

                # spo可以找到在作者
                if (o in zuoqu) and (p in ['歌手', '作词']):
                    zuoqu_new = (s, '作曲', o)
                    zuoqu_add.append(zuoqu_new)
                    special11 += 1

                # spo没找到在作者
                if len(zuoqu_add) == 0:
                    # 书名号可以定位只有一个作曲
                    if len(gequ) == 1:
                        zuoqu_new = (gequ[0][1:-1], '作曲', re.findall('[^是由，、作词填词写歌词、作曲谱曲]+', zuoqu)[0])
                        zuoqu_add.append(zuoqu_new)
                        special11 += 1
        else:
            # 只作曲
            zuoqu_add = []
            zuoqu = re.search('(是|由|是由|，|、)+[^，、]+(作曲|谱曲)', text)
            if zuoqu and ('作曲' not in p_list):
                zuoqu = zuoqu[0]
                for j in line_ensemble_new:
                    s, p, o = j

                    # spo可以找到在作者
                    if (o in zuoqu) and (p in ['歌手']):
                        zuoqu_new = (s, '作曲', o)
                        zuoqu_add.append(zuoqu_new)
                        special11 += 1

                    # spo没找到在作者
                    if len(zuoqu_add) == 0:
                        # 书名号可以定位只有一个作曲
                        if len(gequ) == 1:
                            zuoqu_new = (gequ[0][1:-1], '作曲', re.findall('[^是由，、作曲谱曲]+', zuoqu)[0])
                            zuoqu_add.append(zuoqu_new)
                            special11 += 1
        line_ensemble_new = line_ensemble_new + zuoqu_add
        line_ensemble_new = sorted(list(set(line_ensemble_new)))

        # 规则12 sp相同，o互相包含/po相同，s互相包含（0.0015）
        n1 = 0
        n2 = 0
        spo_remove = []
        sp = {}
        po = {}
        for j in line_ensemble_new:
            s, p, o = j
            spo_n = line_ensemble.count(j)

            if (s, p) in sp:
                if ((o in sp[(s, p)][0]) or (sp[(s, p)][0] in o)) and (o != sp[(s, p)][0]):
                    if sp[(s, p)][1] < spo_n:
                        # 删除频数小的
                        spo_remove.append((s, p, sp[(s, p)][0]))
                        sp[(s, p)] = [o, spo_n]
                    else:
                        spo_remove.append(j)
                    special12_1 += 1
                    n1 += 1
            else:
                sp.update({(s, p): [o, spo_n]})

            if (p, o) in po:
                if ((s in po[(p, o)][0]) or (po[(p, o)][0] in s)) and (s != po[(p, o)][0]):
                    if po[(p, o)][1] < spo_n:
                        spo_remove.append((po[(p, o)][0], p, o))
                        po[(p, o)] = [s, spo_n]
                    else:
                        spo_remove.append(j)
                    special12_2 += 1
                    n2 += 1
            else:
                po.update({(p, o): [s, spo_n]})

        for j in spo_remove:
            line_ensemble_new.remove(j)

        # 对未到阈值的样本再计算一次得分（准确率0.68左右，提升0.0002-0.0005）
        line_ensemble_new_1 = []
        s_list = [j[0] for j in line_ensemble]
        p_list = [j[1] for j in line_ensemble]
        o_list = [j[2] for j in line_ensemble]
        spo_k = find_topk(line_ensemble, 40)
        for j in line_ensemble_new:
            s, p, o = j
            if line_ensemble.count(j) <= 30 and line_ensemble.count(j) > 20:
                j_idx = spo_k.index([j, line_ensemble.count(j)])

                if j_idx > 0:
                    spo_next_n = spo_k[j_idx - 1][1]
                else:
                    spo_next_n = 0.01

                new_line = [len(line_ensemble),
                            s_list.count(s),
                            p_list.count(p),
                            o_list.count(o),
                            j_idx,
                            line_ensemble.count(j),
                            line_ensemble.count(j) / len(line_ensemble),
                            spo_next_n,
                            line_ensemble.count(j) / spo_next_n,
                            1 - j_idx / len(line_ensemble),
                            s_list.count(s) / len(line_ensemble),
                            p_list.count(p) / len(line_ensemble),
                            o_list.count(o) / len(line_ensemble), ]
                score = clf.predict(np.array([new_line]))[0]
                if score == 1:
                    line_ensemble_new_1.append(j)
            else:
                line_ensemble_new_1.append(j)

        line_ensemble_new = line_ensemble_new_1
        line_ensemble_new = sorted(list(set(line_ensemble_new)))


        empty.append(i)
        tpn += len(line_ensemble_new)

        line_result = {"text": text}
        spo_list = [{"object_type": "XX",
                     "predicate": j[1],
                     "object": j[2],
                     "subject_type": "XX",
                     "subject": j[0]} for j in line_ensemble_new]
        line_result["spo_list"] = spo_list
        f.write(json.dumps(line_result, ensure_ascii=False) + '\n')

        f1.write(json.dumps({"text": text,
                             'spo_list': line_ensemble_new}, ensure_ascii=False) + '\n')

    f.close()
    f1.close()
    print('无spo：', len(empty))
    print(tpn)
    print('special:',
          'special1:', special1, '\n',
          'special2:', special2, '\n',
          'special3:', special3, '\n',
          'special4:', special4, '\n',
          'special5:', special5, '\n',
          'special6:', special6, '\n',
          'special7:', special7, '\n',
          'special8:', special8, '\n',
          'special9:', special9, '\n',
          'special10:', special10, '\n',
          'special11:', special11, '\n',
          'special12:', special12_1, special12_2, '\n',
          'special13:', special13, '\n',
          )


if __name__ == '__main__':
    submit(20)
