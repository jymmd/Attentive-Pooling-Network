import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class APN(nn.Module):
    def __init__(self,
                 embed_size, embed_dim, embed_path,
                 encoder_type,
                 rnn_hidden_size, rnn_bidirectional, rnn_num_layer,
                 cnn_num_layer, cnn_kernel_sizes, cnn_num_kernel
                 ):
        super(APN, self).__init__()
        if embed_path:
            embed_matrix = np.load(embed_path)
            self.embed = nn.Embedding.from_pretrained(embed_matrix, freeze=True)
        else:
            self.embed = nn.Embedding(embed_size, embed_dim)
        if encoder_type == 'rnn':
            self.encode = nn.LSTM(input_size=embed_dim, hidden_size=rnn_hidden_size, num_layers=rnn_num_layer,
                                  batch_first=True, bidirectional=rnn_bidirectional)
            if rnn_bidirectional:
                self.U = nn.Parameter(torch.randn(2 * rnn_hidden_size, 2 * rnn_hidden_size))
            else:
                self.U = nn.Parameter(torch.randn(rnn_hidden_size, rnn_hidden_size))
        elif encoder_type == 'cnn':
            pass
        else:
            raise NotImplementedError('NotImplementedError')
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-08)

    def _get_score(self, qu, q_encode, a):
        a_embed = self.embed(a)  # B L E
        a_encode, _ = self.encode(a_embed)  # B L 2H
        G = torch.tanh(qu.bmm(a_encode.transpose(1, 2)))  # B L L
        q_weight = F.softmax(torch.max(G, dim=2)[0], dim=1).unsqueeze(1)  # B 1 L
        q_att = q_weight.bmm(q_encode).squeeze()
        a_weight = F.softmax(torch.max(G, dim=1)[0], dim=1).unsqueeze(1)  # B 1 L
        a_att = a_weight.bmm(a_encode).squeeze()
        return self.cos(q_att, a_att)

    def forward(self, *input):
        if len(input) > 2:
            q, a_pos, a_neg_list = input[0], input[1], input[2:]
            q_embed = self.embed(q)  # B L E
            q_encode, _ = self.encode(q_embed)  # B L 2H
            qu = q_encode.bmm(self.U.unsqueeze(0).expand(q.size(0), -1, -1))
            a_pos_score = self._get_score(qu, q_encode, a_pos)
            a_neg_score_list = torch.cat([self._get_score(qu, q_encode, a_neg).unsqueeze(1) for a_neg in a_neg_list],
                                         dim=1)  # B neg_sample_rate
            a_neg_score_max = torch.max(a_neg_score_list, dim=1)[0]
            return a_pos_score, a_neg_score_max
        else:
            q, a = input[0], input[1]
            q_embed = self.embed(q)  # B L E
            q_encode, _ = self.encode(q_embed)  # B L 2H
            qu = q_encode.bmm(self.U.unsqueeze(0).expand(q.size(0), -1, -1))
            simi_score = self._get_score(qu, q_encode, a)
            return simi_score


if __name__ == '__main__':
    embed_size = 20
    embed_dim = 100
    embed_path = False
    encoder_type = 'rnn'
    rnn_hidden_size = 30
    rnn_bidirectional = True
    rnn_num_layer = 1
    cnn_num_layer = None
    cnn_kernel_sizes = None
    cnn_num_kernel = None
    m = APN(embed_size, embed_dim, embed_path,
            encoder_type,
            rnn_hidden_size, rnn_bidirectional, rnn_num_layer,
            cnn_num_layer, cnn_kernel_sizes, cnn_num_kernel
            )

    # train
    batch_size = 28
    q_len = 22
    a_len = 33
    neg_sample_rate = 10
    q, a_pos, a_neg_list = (torch.randint(embed_size, (batch_size, q_len), dtype=torch.long),
                            torch.randint(embed_size, (batch_size, a_len), dtype=torch.long),
                            [torch.randint(embed_size, (batch_size, a_len), dtype=torch.long)] * neg_sample_rate
                            )
    pos_score, neg_score = m(q, a_pos, *a_neg_list)
    g_pos_score = torch.randn(*pos_score.size())
    g_neg_score = torch.randn(*neg_score.size())
    pos_score.backward(g_pos_score, retain_graph=True)
    neg_score.backward(g_neg_score, retain_graph=True)
    # print(pos_score.size())
    assert pos_score.size() == (batch_size,) == neg_score.size()
    # test
    q, q_test = (torch.randint(embed_size, (batch_size, q_len), dtype=torch.long),
                 torch.randint(embed_size, (batch_size, a_len), dtype=torch.long)
                 )
    simi_score = m(q, q_test)
    # print(simi_score.size())
    assert simi_score.size() == (batch_size,)
    # need a threshold
