import torch
from torch import nn
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
    def __init__(self, vocab_size=38, embed_size=100, hidden_size=256, output_size=38, num_layers=1):
        super(HandInfer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        device = x.device
        mask = x != 30
        lengths = mask.sum(dim=1)
        x = self.embedding(x)
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h0 = torch.zeros(self.num_layers, 4, self.hidden_size, device=device)  # initial hidden state h0
        packed_output, hidden_state = self.rnn(packed_input, h0)
        out, out_lens = pad_packed_sequence(packed_output, batch_first=True)
        predict = self.fc(out[:, -1, :])  # use last hidden state
        # predict = self.fc(out)  # use all hidden state
        # TODO: argmax can't backward(), it runs but network does not update, consider encoder-decoder architecture
        predict_tile = predict.argmax(1)

        predict_seq = [[], [], [], []]
        for sublist, elem in zip(predict_seq, predict_tile):
            sublist.append(elem)
        # TODO: use teacher forcing technique to boost training
        for i in range(12):
            predict_embed = self.embedding(predict_tile.unsqueeze(1))
            predict_next, hidden_state = self.rnn(predict_embed, hidden_state)
            predict = self.fc(predict_next[:, -1, :])
            predict_tile = predict.argmax(1)
            for sublist, elem in zip(predict_seq, predict_tile):
                sublist.append(elem)

        for i in range(4):
            predict_seq[i] = data_to_code(predict_seq[i])
        result = torch.tensor(predict_seq, dtype=torch.float, device=device, requires_grad=True)
        return result
