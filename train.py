import torch
import torch.optim as optim
from net import DatasetRNN, collate_fn_RNN_pos
from net import Match_LSTM
import pickle

import os
import pandas as pd

DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(1)

EMBEDDING_DIM = 300
pos_dim = 50
num_layers = 2
HIDDEN_DIM = 512
BATCH_SIZE = 64
num_words = 8000

Net = Match_LSTM

# ['baidubaike', 'renmin', 'sogounews', 'weibo', 'wiki', 'zhihu']
with open(DIR + '/data_deal/%d/weight_baidubaike.pkl' % num_words, 'rb') as f:
    weight = pickle.load(f)
    weight = torch.FloatTensor(weight).to(device)

# 导入预处理标签
with open(DIR + '/data_deal/p.pkl', 'rb') as f:
    p_index, index_p = pickle.load(f)

# 导入文本编码、词典
with open(DIR + '/data_deal/%d/word_index.pkl' % num_words, 'rb') as f:
    word_index = pickle.load(f)
with open(DIR + '/data_deal/pos_index.pkl', 'rb') as f:
    pos_index = pickle.load(f)

with open(DIR + '/data_deal/%d/train_data_process.pkl' % num_words, 'rb') as f:
    train_data_process = pickle.load(f)
with open(DIR + '/data_deal/%d/dev_data_process.pkl' % num_words, 'rb') as f:
    dev_data_process = pickle.load(f)
with open(DIR + '/data_deal/%d/test1_data_process.pkl' % num_words, 'rb') as f:
    test1_data_process = pickle.load(f)

trainloader = torch.utils.data.DataLoader(
    dataset=DatasetRNN(train_data_process),
    batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_RNN_pos)

devloader = torch.utils.data.DataLoader(
    dataset=DatasetRNN(dev_data_process),
    batch_size=1, shuffle=False, collate_fn=collate_fn_RNN_pos)
spo_lists_labels = [i['spo_list_raw'] for i in dev_data_process]


def train(epochs=5, mask=True):
    # vocab_size还有pad和unknow，要+2
    model = Net(vocab_size=len(word_index) + 2,
                pos_size=len(pos_index) + 2,
                tag_to_ix=p_index,
                embedding_dim=EMBEDDING_DIM,
                pos_dim=pos_dim,
                num_layers=num_layers,
                hidden_dim=HIDDEN_DIM,
                mask=mask,
                s_weight=2.5,
                o_weight=1,
                weight=weight,
                device=device).to(device)
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    score = []
    for epoch in range(epochs):
        print('Start Epoch: %d\n' % (epoch + 1))
        sum_loss = 0.0
        model.train()
        for i, data in enumerate(trainloader):
            model.zero_grad()
            text_seqs, pos_seqs, s_B_labels, s_E_labels, s_B_E_ins, o_B_labels, o_E_labels = data
            text_seqs = text_seqs.to(device)
            pos_seqs = pos_seqs.to(device)
            s_B_labels = s_B_labels.to(device)
            s_E_labels = s_E_labels.to(device)
            s_B_E_ins = s_B_E_ins.to(device)
            o_B_labels = o_B_labels.to(device)
            o_E_labels = o_E_labels.to(device)

            # 损失函数
            loss = model.cal_total_loss(text_seqs, pos_seqs, s_B_labels, s_E_labels,
                                        s_B_E_ins, o_B_labels, o_E_labels)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            if (i + 1) % 100 == 0:
                print('\nEpoch: %d ,batch: %d, loss = %f' % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0

        # 不调试forward保存建议整个网络，调试的话建议保存网络参数
        torch.save(model, './models/%d/%03d.pth' % (num_words, epoch + 1))

        # dev得分
        model.eval()
        p_len = 0.001
        l_len = 0.001
        correct_len = 0.001
        spo_list_all = []
        for idx, data in enumerate(devloader):
            text_seqs, pos_seqs, s_B_labels, s_E_labels, s_B_E_ins, o_B_labels, o_E_labels = data
            text_seqs = text_seqs.to(device)
            pos_seqs = pos_seqs.to(device)

            spo_list = model(text_seqs,
                             pos_seqs,
                             dev_data_process[idx]['text'],
                             index_p)
            spo_list_all.append(spo_list)
            set_p = set(spo_list)
            set_l = set(spo_lists_labels[idx])
            p_len += len(set_p)
            l_len += len(set_l)
            correct_len += len(set_p.intersection(set_l))
            if (idx + 1) % 1000 == 0:
                print('finish dev %d' % (idx + 1))
        Precision = correct_len / p_len
        Recall = correct_len / l_len
        F1 = 2 * Precision * Recall / (Precision + Recall)
        score.append([epoch + 1, Precision, Recall, F1])
        print('\nEpoch: %d ,Precision:%f, Recall:%f, F1:%f' % (epoch + 1, Precision, Recall, F1))

        score_df = pd.DataFrame(score, columns=['Epoch', 'Precision', 'Recall', 'F1'])
        print(score_df)
        score_df.to_csv('./models/%d/dev.csv' % (num_words), index=False)
        with open('./models/%d/dev_%03d.pkl' % (num_words, epoch + 1), 'wb') as f:
            pickle.dump(spo_list_all, f)

        # test1
        model.eval()
        spo_list_all = []
        for idx, i in enumerate(test1_data_process):
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
        with open('./models/%d/test1_%03d.pkl' % (num_words, epoch + 1), 'wb') as f:
            pickle.dump(spo_list_all, f)


if __name__ == '__main__':
    train(epochs=20, mask=True)
