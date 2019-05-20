import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Features(nn.Module):
    def __init__(self,
                 vocab_size,
                 pos_size,
                 embedding_dim=256,
                 pos_dim=None,
                 num_layers=3,
                 hidden_dim=256,
                 mask=False,
                 weight=None,
                 device='cpu'):
        super(Features, self).__init__()
        self.device = device
        # 词嵌入
        self.vocab_size = vocab_size
        self.pos_size = pos_size

        if weight is None:
            self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        else:
            embedding_dim = weight.shape[1]
            self.word_embeds = nn.Embedding.from_pretrained(weight, freeze=True)
        self.embedding_dim = embedding_dim

        # 文本信息
        self.mask = mask
        self.hidden_dim = hidden_dim

        self.pos_dim = pos_dim
        if pos_dim:
            self.pos_embeds = nn.Embedding(pos_size, pos_dim)
            self.lstm = nn.LSTM(embedding_dim + pos_dim, hidden_dim // 2,
                                num_layers=num_layers, bidirectional=True,
                                batch_first=True)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                                num_layers=num_layers, bidirectional=True,
                                batch_first=True)

    def forward(self, sentence_seqs, pos_seqs):
        if self.pos_dim:
            embeds = torch.cat([self.word_embeds(sentence_seqs), self.pos_embeds(pos_seqs)], dim=-1)
        else:
            embeds = self.word_embeds(sentence_seqs)
        if self.mask:
            mask_idx = 1 - torch.eq(sentence_seqs, 0)
            self.mask_idx = mask_idx
            lengths = mask_idx.sum(dim=1)
            embeds = nn.utils.rnn.pack_padded_sequence(input=embeds,
                                                       lengths=lengths,
                                                       batch_first=True)
            sentence_features, self.hidden = self.lstm(embeds)
            sentence_features = nn.utils.rnn.pad_packed_sequence(sequence=sentence_features,
                                                                 batch_first=True,
                                                                 padding_value=0)[0]
        else:
            self.mask_idx = torch.ones(sentence_seqs.size()).to(device)
            sentence_features, self.hidden = self.lstm(embeds)

        return sentence_features, mask_idx
