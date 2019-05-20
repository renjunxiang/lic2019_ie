import torch
import torch.nn as nn
from .task import Locate_Subject, Locate_Object
from .module import Features


class Match_LSTM(nn.Module):
    """
    1.输入 sentence，预测  subject的位置
        sigmoid预测首尾位置
    2.输入 sentence+subject的位置，预测 object的位置+subject-object的关系
        softmax预测首尾位置+关系类别
    3.计算1和2的loss
    """

    def __init__(self,
                 vocab_size,
                 pos_size,
                 tag_to_ix,
                 embedding_dim=256,
                 pos_dim=None,
                 num_layers=3,
                 hidden_dim=256,
                 mask=False,
                 s_weight=2.5,
                 o_weight=1,
                 weight=None,
                 device='cpu', ):
        super(Match_LSTM, self).__init__()

        # self.tagset_size = len(tag_to_ix)
        self.device = device
        self.s_weight = s_weight

        # 词嵌入
        self._get_sentence_features = Features(vocab_size,
                                               pos_size,
                                               embedding_dim,
                                               pos_dim,
                                               num_layers,
                                               hidden_dim,
                                               mask,
                                               weight,
                                               device)

        # subject起止位置，只做二分类
        self._get_s_position = Locate_Subject(hidden_dim)

        # 拼接subject起止位置，还需要带上关系类别，是多分类
        self._get_o_position = Locate_Object(tag_to_ix, hidden_dim, o_weight, device)

    def cal_total_loss(self, sentence_seqs, pos_seqs, s_B_labels, s_E_labels,
                       s_B_E_ins, o_B_labels, o_E_labels):
        """
        计算损失总和
        :param sentence_seqs:
        :param s_B_labels:
        :param s_E_labels:
        :param s_B_in:
        :param s_E_in:
        :param o_B_labels:
        :param o_E_labels:
        :return:
        """
        # 计算文本语义
        sentence_features, mask_idx = self._get_sentence_features(sentence_seqs, pos_seqs)

        # 预测subject的起止
        s_B_idxs, s_E_idxs = self._get_s_position(sentence_features)

        # 预测object的起止
        o_B_idxs, o_E_idxs = self._get_o_position(sentence_features, s_B_E_ins)

        mask_idx = mask_idx.float()

        # 计算subject的损失,去除mask部分
        loss1 = self._get_s_position.cal_loss(s_B_idxs, s_E_idxs,
                                              s_B_labels, s_E_labels,
                                              mask_idx)

        # 计算object的损失,reshape至[-1,tagsize],计算交叉熵,去除mask部分
        loss2 = self._get_o_position.cal_loss(o_B_idxs, o_E_idxs,
                                              o_B_labels, o_E_labels,
                                              mask_idx)

        return self.s_weight * loss1 + loss2

    def forward(self, sentence_seqs, pos_seqs, sentences, index_p):
        """
        一条条预测吧,好像没法一起做
        :param sentence_seqs:
        :return:
        """
        spo_list = []
        sentence_features, _ = self._get_sentence_features(sentence_seqs, pos_seqs)
        # 预测subject的起止
        s_B_idxs, s_E_idxs = self._get_s_position(sentence_features)
        s_B_idxs = nn.Sigmoid()(s_B_idxs[0]) > 0.5
        s_E_idxs = nn.Sigmoid()(s_E_idxs[0]) > 0.5
        for s_B_idx, s_B_score in enumerate(s_B_idxs):
            if s_B_score == 1:
                s_B_E = []
                # E是在B之后的,索引从B开始
                for s_E_idx, s_E_score in enumerate(s_E_idxs[s_B_idx:]):
                    if s_E_score == 1:
                        s_B_E = [s_B_idx, s_B_idx + s_E_idx]
                        break
                # print(s_B_E)
                # 如果存在subject,再去找object
                if s_B_E:
                    o_B_idxs, o_E_idxs = self._get_o_position(sentence_features,
                                                              torch.LongTensor([s_B_E]).to(self.device))
                    # 每个字概率最大的标签
                    o_B_idxs = o_B_idxs[0].argmax(dim=1).tolist()
                    o_E_idxs = o_E_idxs[0].argmax(dim=1).tolist()
                    # print(o_B_idxs)
                    # print(o_E_idxs)
                    for o_B_idx, o_B_score in enumerate(o_B_idxs):
                        if o_B_score > 0:
                            for o_E_idx, o_E_score in enumerate(o_E_idxs[o_B_idx:]):
                                # 当起止标签一致时，判定为object
                                if o_B_score == o_E_score:
                                    spo_list.append((sentences[s_B_E[0]:(s_B_E[1] + 1)],
                                                     index_p[o_B_score],
                                                     sentences[o_B_idx:(o_B_idx + o_E_idx + 1)]))
                                    break

        return spo_list
