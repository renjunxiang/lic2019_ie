import pickle
import time
import numpy as np
from collections import OrderedDict
import os

DIR = os.path.dirname(os.path.abspath(__file__))


def get_embdding(embedding_name='baidubaike', num_words=8000):
    path = './Chinese-Word-Vectors/sgns.%s.bigram-char' % embedding_name

    with open(DIR + '/data_deal/%d/word_index.pkl' % num_words, 'rb') as f:
        word_index = pickle.load(f)

    if os.path.exists(DIR + '/data_deal/char_embedding_%s.pkl' % embedding_name):
        with open(DIR + '/data_deal/char_embedding_%s.pkl' % embedding_name, 'rb') as f:
            char_embedding = pickle.load(f)
    else:
        # 抽取字向量
        def get_char_embedding(path):
            char_embedding = OrderedDict()
            f = open(path, 'r', encoding='utf-8')
            lines = f.readlines()[1:]
            for line in lines:
                line = line.split(' ')
                if len(line[0]) == 1:
                    char_embedding[line[0]] = np.array(line[1:-1], dtype=float)

            return char_embedding

        char_embedding = get_char_embedding(path)
        with open(DIR + '/data_deal/char_embedding_%s.pkl' % embedding_name, 'wb') as f:
            pickle.dump(char_embedding, f)

    # 存储为网络embedding层权重
    def embedding(char_embedding, word_index):
        weight = np.zeros([num_words + 2, 300])
        for word, idx in word_index.items():
            if word in char_embedding:
                weight[idx] = char_embedding[word]

        return weight

    weight = embedding(char_embedding, word_index)
    print(weight.shape)
    with open(DIR + '/data_deal/%d/weight_%s.pkl' % (num_words, embedding_name), 'wb') as f:
        pickle.dump(weight, f)


if __name__ == '__main__':
    embedding_name_list = ['baidubaike', 'renmin', 'sogounews', 'weibo', 'wiki', 'zhihu']
    num_words_list = [6000, 7000]
    for embedding_name in embedding_name_list:
        for num_words in num_words_list:
            print('start %s %d' % (embedding_name, num_words))
            get_embdding(embedding_name, num_words)
            print('finish %s %d' % (embedding_name, num_words))
