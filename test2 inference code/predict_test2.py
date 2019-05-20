import codecs
import torch
import pickle
import os
import json

DIR = os.path.dirname(os.path.abspath(__file__))
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open(DIR + '/data_deal/p.pkl', 'rb') as f:
    p_index, index_p = pickle.load(f)

file_name = {
    6000: {
        'baidubaike': {
            'match_4_lstm_768_50_4_1_2': [
                6,  # 0.789
            ],
            'match_4_lstm_2048_50_4_1_2': [
                6,  # 0.790
                4, 9,  # 0.788
            ],
            'match_4_lstm_2048_60_4_1_2': [  # 新
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
            'match_4_lstm_768_50_4_1_2': [
                7,  # 0.789
            ],
            'match_4_lstm_768_100_4_1_2': [
                8,  # 0.789
                9, 7,  # 0.788
            ],
            'match_4_lstm_2048_30_4_1_2': [
                10,  # 0.790
                8, 7, 6,  # 0.788
            ],
            'match_4_lstm_2048_50_4_1_2': [
                6, 5, 10, 8,  # 0.790
                9,  # 0.788
            ],
            'match_4_lstm_2048_60_4_1_2': [  # 新
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
for num_words in file_name:
    # 读取test2预处理
    with open(DIR + '/data_deal/%d/test2_data_process.pkl' % num_words, 'rb') as f:
        test2_data_process = pickle.load(f)

    for embedding_name in file_name[num_words]:
        for model_name in file_name[num_words][embedding_name]:
            for model_idx in file_name[num_words][embedding_name][model_name]:
                model = torch.load('/home/disk0/renjunxiang/baidu/trunk/models/%d/%s/%s/%03d.pth' % (
                num_words, embedding_name, model_name, model_idx))
                print('finish load:/%d/%s/%s/%03d.pth' % (num_words, embedding_name, model_name, model_idx))
                device = model.device

                # 预测test2
                model.eval()
                spo_list_all = []
                for idx, i in enumerate(test2_data_process):
                    text = i['text']
                    text_seqs = [i['text_seq']]
                    pos_seqs = [i['pos_seq']]

                    text_seqs = torch.LongTensor(text_seqs).to(device)
                    pos_seqs = torch.LongTensor(pos_seqs).to(device)

                    spo_list = model(text_seqs,
                                     pos_seqs,
                                     text,
                                     index_p)
                    spo_list_all.append(spo_list)
                with open('./test2/%d/%s/%s/test2_%03d.pkl' % (num_words, embedding_name, model_name, model_idx),
                          'wb') as f:
                    pickle.dump(spo_list_all, f)
                    print('finish: /%d/%s/%s/%03d.pkl' % (num_words, embedding_name, model_name, model_idx))
