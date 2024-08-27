import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class HandInfer(nn.Module):
    def __init__(self, vocab_size=38, embed_size=100, hidden_size=256, output_size=190, num_layers=2):
        super(HandInfer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        mask = x != 30
        lengths = mask.sum(dim=1)
        x = self.embedding(x)
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h0 = torch.zeros(self.num_layers, 4, self.hidden_size).to(x.device)  # initial hidden state h0
        packed_output, _ = self.rnn(packed_input, h0)
        out, _ = pad_packed_sequence(packed_output, batch_first=True)
        out = self.fc(out[:, -1, :])
        return out
