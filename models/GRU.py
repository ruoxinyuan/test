import torch
import torch.nn as nn

class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, lengths):
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)

        if isinstance(lengths, int):
            sorted_lengths = torch.tensor([lengths]).to(x.device)
            sorted_x = x
        else:
            # 排序输入和长度，以便最长的序列是第一个
            sorted_lengths, sort_idx = lengths.sort(descending=True)
            sorted_x = x[sort_idx]

        packed_x = nn.utils.rnn.pack_padded_sequence(sorted_x, sorted_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_x, h0)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # 获取每个序列的最后一个有效时间步的索引
        last_indices = (output_lengths - 1).long()

        # 直接使用索引获取最后一个时间步的输出
        last_output = output[torch.arange(output.size(0)), last_indices, :]

        if not isinstance(lengths, int):
            # 由于我们对输入进行了排序，现在需要恢复原始顺序
            _, original_idx = sort_idx.sort()
            last_output = last_output[original_idx]

        last_output = self.layer_norm(last_output)
        last_output = self.dropout(last_output)
        
        logits = self.fc(last_output)
        return logits
