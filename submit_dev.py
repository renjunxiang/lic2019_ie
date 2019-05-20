import torch
import pickle
import os

mask = True
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
with open(DIR + '/data_deal/%d/dev_data_process.pkl' % num_words, 'rb') as f:
    dev_data_process = pickle.load(f)

# 导入预处理标签
with open(DIR + '/data_deal/p.pkl', 'rb') as f:
    p_index, index_p = pickle.load(f)

p_count_text = {i: 0 for i in p_index}
p_count_spo = {i: 0 for i in p_index}

dev_predicts = []

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
                with open('./models_old/%d/%s/%s/dev_%03d.pkl' % (num_words, embedding_name, model_name, model_idx),
                          'rb') as f:
                    dev = pickle.load(f)
                    dev_predicts.append(dev)

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
                with open('./models/%d/%s/%s/dev_%03d.pkl' % (num_words, embedding_name, model_name, model_idx),
                          'rb') as f:
                    dev = pickle.load(f)
                    dev_predicts.append(dev)


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

    return v[:k]


def dev_score(n):
    dev_ensemble = []
    p_len = 0.001
    l_len = 0.001
    correct_len = 0.001
    for i in range(len(dev_data_process)):
        set_l = set(dev_data_process[i]['spo_list_raw'])

        line_ensemble = []
        for dev_predict in dev_predicts:
            line_ensemble += dev_predict
        line_ensemble_set = set(line_ensemble)
        line_ensemble_new = [j for j in line_ensemble_set if line_ensemble.count(j) > n]

        dev_ensemble.append(line_ensemble_new)
        set_p = set(line_ensemble_new)

        correct_len += len(set_p.intersection(set_l))
        p_len += len(set_p)
        l_len += len(set_l)

    Precision = correct_len / p_len
    Recall = correct_len / l_len
    F1 = 2 * Precision * Recall / (Precision + Recall)
    print('本地, Precision:%f, Recall:%f, F1:%f' % (Precision, Recall, F1))

    Precision += 0.08
    Recall -= 0.015
    F1 = 2 * Precision * Recall / (Precision + Recall)
    print('预计, Precision:%f, Recall:%f, F1:%f' % (Precision, Recall, F1))

    spo_stat = {}
    for j in p_count_spo:
        spo_stat.update({j: [p_count_spo[j], p_count_text[j]]})


for i in range(20, 31):
    print(i)
    dev_score(i)
