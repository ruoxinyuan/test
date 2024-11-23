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
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)

        if isinstance(lengths, int):
            # Handle the case where lengths is a single integer
            sorted_lengths = torch.tensor([lengths]).to(x.device)
            sorted_x = x
        else:
            # Sort input and lengths so that the longest sequence is first
            sorted_lengths, sort_idx = lengths.sort(descending=True)
            sorted_x = x[sort_idx]

        # Pack the input tensor to handle variable-length sequences
        packed_x = nn.utils.rnn.pack_padded_sequence(sorted_x, sorted_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_x, h0)
        
        # Unpack the output back to padded tensor format
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Get the index of the last valid time step for each sequence
        last_indices = (output_lengths - 1).long()

        # Retrieve the output corresponding to the last valid time step
        last_output = output[torch.arange(output.size(0)), last_indices, :]

        if not isinstance(lengths, int):
            # Restore the original order of sequences
            _, original_idx = sort_idx.sort()
            last_output = last_output[original_idx]

        # Apply layer normalization to the last output
        last_output = self.layer_norm(last_output)
        # Apply dropout for regularization
        last_output = self.dropout(last_output)
        
        # Pass the output through a fully connected layer to produce logits
        logits = self.fc(last_output)
        return logits
