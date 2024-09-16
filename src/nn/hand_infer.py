import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from src.utils.data_process import data_to_code


class HandInfer(nn.Module):
    """
    逐步生成（递归式预测）：
    1.输入初始序列：你先输入一个初始序列，比如一句话的开头。
    2.预测下一个词：RNN 会根据输入序列预测下一个词。
    3.将预测结果作为下一个时间步的输入：将这个预测的词作为下一次时间步的输入，递归地进行下去。
    4.重复生成：这个过程一直重复，直到生成完整的序列，达到所需的长度或者遇到某个停止条件（如生成句子结束的标记）。

    Teacher Forcing：一种常见的训练方法是 teacher forcing，即在训练时，模型的输入不仅是前一个时间步的预测，
    还可以使用真实的目标值作为输入，从而加速收敛。Teacher Forcing 的做法是在训练阶段，模型在每个时间步上
    将前一时间步的真实标签作为输入，而不是模型的预测结果。这减少了累积的预测误差对训练过程的影响。
    """
    def __init__(self, one_hot_size=38, embed_size=50, hidden_size=128, num_layers=1):
        super(HandInfer, self).__init__()
        self.batch_size = 4
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(one_hot_size, embed_size)
        self.encoder_rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.decoder_rnn = nn.RNN(one_hot_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, one_hot_size)

    def forward(self, x, y=None):
        device = x.device
        mask = x != 30
        lengths = mask.sum(dim=1)
        x = self.embedding(x)
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=device)  # initial hidden state h0
        packed_output, hidden_state = self.encoder_rnn(packed_input, h0)  # hidden_state(num_layer, 4, 128)
        encoder_out, encoder_out_lens = pad_packed_sequence(packed_output, batch_first=True)  # encoder_out(4, n, 128)
        encoder_predict = self.fc(encoder_out[:, -1, :])
        predict_tile = F.gumbel_softmax(encoder_predict, tau=1.0, hard=True)

        predict_seq = [[], [], [], []]
        for j in range(13):
            for i, tile in enumerate(predict_tile):
                predict_seq[i].append(tile)
            if (y is not None):
                # use teacher forcing technique to boost training
                predict_tile = y[:, j, :]
            predict_tile = predict_tile.unsqueeze(1)
            predict_next, hidden_state = self.decoder_rnn(predict_tile, hidden_state)
            predict_next = self.fc(predict_next[:, -1, :])
            predict_tile = F.gumbel_softmax(predict_next, tau=1.0, hard=True)
        for i in range(self.batch_size):
            predict_seq[i] = torch.stack(predict_seq[i], dim=0)
        predict_seq = torch.stack(predict_seq, dim=0)

        return predict_seq
