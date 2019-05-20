import torch
from torch.utils.data import Dataset
from random import choice

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义数据读取方式
class DatasetRNN(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


def collate_fn_RNN_pos(batch):
    text_seqs = []
    pos_seqs = []
    s_start_labels = []
    s_end_labels = []
    s_start_end_in = []
    o_start_labels = []
    o_end_labels = []
    batch_len = []

    # 列表必须做拷贝,否则原始数据也会发生改变
    for i in batch:
        text_seqs.append(i['text_seq'][:])
        pos_seqs.append(i['pos_seq'][:])
        s_start_labels.append(i['s_start_label'][:])
        s_end_labels.append(i['s_end_label'][:])

        # 端到端一次只能抽样训练一组spo
        spo = choice(i['spo_list'])
        s_start_end_in.append(spo[0][:])
        o_start_labels.append(spo[1][:])
        o_end_labels.append(spo[2][:])
        batch_len.append(len(i['text_seq']))
    len_max = max(batch_len)

    # 按文本长度从大到小排序（不用nn.utils.rnn.pack_padded_sequence则不需要排序）
    idx = sorted(range(len(batch)), key=lambda x: batch_len[x], reverse=True)
    text_seqs = [text_seqs[i] for i in idx]
    pos_seqs = [pos_seqs[i] for i in idx]
    s_start_labels = [s_start_labels[i] for i in idx]
    s_end_labels = [s_end_labels[i] for i in idx]
    s_start_end_in = [s_start_end_in[i] for i in idx]
    o_start_labels = [o_start_labels[i] for i in idx]
    o_end_labels = [o_end_labels[i] for i in idx]
    batch_len = [batch_len[i] for i in idx]

    # 填充<PAD>的编码和标注0
    for i in range(len(batch)):
        text_seqs[i] += [0] * (len_max - batch_len[i])
        pos_seqs[i] += [0] * (len_max - batch_len[i])
        s_start_labels[i] += [0] * (len_max - batch_len[i])
        s_end_labels[i] += [0] * (len_max - batch_len[i])
        o_start_labels[i] += [0] * (len_max - batch_len[i])
        o_end_labels[i] += [0] * (len_max - batch_len[i])

    text_seqs = torch.LongTensor(text_seqs)
    pos_seqs = torch.LongTensor(pos_seqs)
    s_start_labels = torch.Tensor(s_start_labels)
    s_end_labels = torch.Tensor(s_end_labels)
    s_start_end_in = torch.LongTensor(s_start_end_in)
    o_start_labels = torch.LongTensor(o_start_labels)
    o_end_labels = torch.LongTensor(o_end_labels)

    return text_seqs, pos_seqs, s_start_labels, s_end_labels, s_start_end_in, o_start_labels, o_end_labels
