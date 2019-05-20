import torch
import torch.nn as nn
from torchcrf import CRF

crf_tag2id = {
    'P': 0,
    'START_TAG': 1,
    'STOP_TAG': 2,
    'U': 3,
    'B': 4, 'M': 5, 'E': 6,
    'S': 7,
}

crf_id2tag = {j: i for i, j in crf_tag2id.items()}


class Locate_Subject(nn.Module):
    """
    1.输入 sentence，预测 sentence的序列标注
    """

    def __init__(self,
                 hidden_dim=256):
        super(Locate_Subject, self).__init__()

        # subject起止位置，只做二分类
        self.crf = CRF(self.crf_size, batch_first=True)

        # subject起止位置，crf得到
        self.s_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU()
        )
        self.predict_s_crf = nn.Sequential(
            nn.Linear(hidden_dim, self.crf_size)
        )

    def cal_loss(self, sentence_features, tags, mask):
        """
        损失函数=所有序列得分-正确序列得分
        :param sentence:
        :param tags:
        :return:
        """
        sentence_features = sentence_features.transpose(2, 1)
        sentence_features = self.s_conv(sentence_features)
        sentence_features = sentence_features.transpose(2, 1)
        sentence_features = self.predict_s_crf(sentence_features)
        if mask:
            loss = -self.crf(sentence_features, tags, self.mask_idx, reduction='mean')
        else:
            loss = -self.crf(sentence_features, tags, reduction='mean')

        return loss

    def forward(self, sentence_features):
        """
        维特比算法寻找subject的最大得分序列，用于推断
        :param batch_feats:
        :return:
        """
        sentence_features = sentence_features.transpose(2, 1)
        sentence_features = self.s_conv(sentence_features)
        sentence_features = sentence_features.transpose(2, 1)
        sentence_features = self.predict_s_crf(sentence_features)
        best_path = self.crf.decode(sentence_features)

        return best_path
