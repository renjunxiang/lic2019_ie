import torch
import torch.nn as nn


class Locate_Subject(nn.Module):
    """
    输入 sentence，预测 subject的首尾位置
    """

    def __init__(self,
                 hidden_dim=256):
        super(Locate_Subject, self).__init__()

        # subject起止位置，只做二分类
        self.s_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU()
        )
        self.predict_s_B = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        self.predict_s_E = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )

    def cal_loss(self,
                 s_B_idxs,
                 s_E_idxs,
                 s_B_labels,
                 s_E_labels,
                 mask_idx):
        mask_idx = mask_idx.float()

        # 计算subject_B的损失,去除mask部分
        loss1 = nn.BCEWithLogitsLoss(reduce=False)(s_B_idxs, s_B_labels)
        loss1 = (loss1 * mask_idx).sum() / mask_idx.sum()

        # 计算subject_E的损失,去除mask部分
        loss2 = nn.BCEWithLogitsLoss(reduce=False)(s_E_idxs, s_E_labels)
        loss2 = (loss2 * mask_idx).sum() / mask_idx.sum()

        return loss1 + loss2

    def forward(self, sentence_features):
        """
        获得subject的位置
        :param features:
        :return:
        """
        sentence_features = sentence_features.transpose(2, 1)
        sentence_features = self.s_conv(sentence_features)
        sentence_features = sentence_features.transpose(2, 1)
        s_B_idxs = self.predict_s_B(sentence_features)
        s_E_idxs = self.predict_s_E(sentence_features)

        return s_B_idxs.squeeze(-1), s_E_idxs.squeeze(-1)
