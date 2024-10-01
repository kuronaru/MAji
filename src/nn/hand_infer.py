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

    N = batch size
    L = sequence length
    D = 2 if bidirectional=True otherwise 1
    H_in = input_size
    H_out = hidden_size
    """
    def __init__(self, one_hot_size=38, embed_size=50, hidden_size=128, num_layers=1):
        super(HandInfer, self).__init__()
        self.batch_size = 4
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(one_hot_size, embed_size)
        self.encoder = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.GRU(one_hot_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, one_hot_size)
        self.fc2 = nn.Linear(hidden_size, one_hot_size)

    def forward(self, x, y=None):
        device = x.device
        mask = x != 30
        lengths = mask.sum(dim=1)
        x = self.embedding(x)
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=device)  # initial hidden state h0
        packed_output, encoder_hidden = self.encoder(packed_input, h0)  # hidden_state(num_layers, N, H_out)

        output_list = []
        start_token = torch.ones(4, 38, device=device) * 30
        decoder_input = start_token
        decoder_hidden = encoder_hidden
        for j in range(13):
            decoder_input = decoder_input.unsqueeze(1)  # decoder_input(N, 1, H_in)
            decoder_input, decoder_hidden = self.decoder(decoder_input, decoder_hidden)  # decoder_input(N, 1, H_out)
            decoder_input = self.fc2(decoder_input[:, -1, :])
            output_list.append(decoder_input)
            if (y is not None):
                # use teacher forcing technique to boost training
                decoder_input = y[:, j, :]
        output_list = torch.stack(output_list)  # output_list(13, N, H_in)
        outputs = output_list.permute(1, 0, 2)  # outputs(N, 13, H_in)

        return outputs
