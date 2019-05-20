import torch
import torch.nn as nn


class Locate_Object(nn.Module):
    """
    1.输入 sentence，预测  subject的位置
        sigmoid预测首尾位置
    2.输入 sentence+subject的位置，预测 object的位置+subject-object的关系
        softmax预测首尾位置+关系类别
    3.计算1和2的loss
    """

    def __init__(self,
                 tag_to_ix,
                 hidden_dim=256,
                 o_weight=2,
                 device='cpu'):
        super(Locate_Object, self).__init__()
        self.device = device

        self.tagset_size = len(tag_to_ix)

        # 拼接subject起止位置，还需要带上关系类别，是多分类
        self.o_conv = nn.Sequential(
            nn.Conv1d(hidden_dim * 3, hidden_dim, 3, padding=1),
            nn.ReLU()
        )
        self.predict_o_B = nn.Sequential(
            nn.Linear(hidden_dim * 1, self.tagset_size + 1)
        )
        self.predict_o_E = nn.Sequential(
            nn.Linear(hidden_dim * 1, self.tagset_size + 1)
        )
        self.o_weight=o_weight

    def _slice(self, batch_input, batch_slice):
        """
        1.从batch_input做切片,取batch_slice作为索引
        2.拼接回batch_input的hidden_dim上
        :param
        batch_input: [batch_size, time_step, hidden_dim]
        :param
        batch_slice: [batch_size, 2]
        :return:

        batch_input = torch.Tensor([
            [[1, 2], [2, 3], [3, 4]],
            [[2, 3], [3, 4], [4, 5]],
            [[3, 4], [4, 5], [5, 6]]
            ])
        batch_slice = torch.LongTensor([
            [0, 1],
            [1, 2],
            [0, 2]
            ])
        return = torch.Tensor([
            [[1, 2, 1, 2, 2, 3], [2, 3, 1, 2, 2, 3], [3, 4, 1, 2, 2, 3]],
            [[2, 3, 3, 4, 4, 5], [3, 4, 3, 4, 4, 5], [4, 5, 3, 4, 4, 5]],
            [[3, 4, 3, 4, 5, 6], [4, 5, 3, 4, 5, 6], [5, 6, 3, 4, 5, 6]]
            ])
        """
        shape_input = batch_input.size()
        batch_slice = batch_slice.long().unsqueeze(2).expand(-1, -1, shape_input[2])
        tensor_slice = torch.gather(batch_input, 1, batch_slice)
        tensor_slice = tensor_slice.view([shape_input[0], -1]).unsqueeze(1)
        tensor_slice = tensor_slice.expand((-1, shape_input[1], -1))
        tensor_gather = torch.cat([batch_input, tensor_slice], 2)

        return tensor_gather

    def cal_loss(self,
                 o_B_idxs,
                 o_E_idxs,
                 o_B_labels,
                 o_E_labels,
                 mask_idx):
        mask_idx = mask_idx.float()
        batch_shape = mask_idx.size()

        # recall比较低,原因是object识别不出来,增加正样本损失权重
        weight = torch.Tensor([1] + [self.o_weight] * 49).to(self.device)

        # 计算object_B的多分类交叉熵,去除mask部分
        loss3 = nn.CrossEntropyLoss(reduce=False, weight=weight)(o_B_idxs.transpose(2, 1), o_B_labels.long())
        loss3 = (loss3 * mask_idx).sum() / mask_idx.sum()

        # 计算object_E的多分类交叉熵,去除mask部分
        loss4 = nn.CrossEntropyLoss(reduce=False, weight=weight)(o_E_idxs.transpose(2, 1), o_E_labels.long())
        loss4 = (loss4 * mask_idx).sum() / mask_idx.sum()

        return loss3 + loss4

    def forward(self, sentence_features, s_B_Es):
        """
        获得object的位置，subject-object的关系
        :param sentence_features: 文本语义
        :param s_B_Es: 主体起止位置
        :return:
        """

        features_add_s = self._slice(sentence_features, s_B_Es)
        features_add_s = features_add_s.transpose(2, 1)
        features_add_s = self.o_conv(features_add_s)
        features_add_s = features_add_s.transpose(2, 1)
        o_B_idxs = self.predict_o_B(features_add_s)
        o_E_idxs = self.predict_o_E(features_add_s)

        return o_B_idxs, o_E_idxs
