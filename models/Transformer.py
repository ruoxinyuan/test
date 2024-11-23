import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dim_feedforward, max_seq_len):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.position_encoding = nn.Parameter(torch.zeros(1, max_seq_len, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.fc = nn.Linear(model_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # 使用Xavier均匀初始化权重
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def create_attention_mask(self, lengths, max_len):
        if isinstance(lengths, int):
            lengths = torch.tensor([lengths]).to(device)
            
        attention_mask = torch.zeros(len(lengths), max_len, dtype=torch.bool)

        for i, length in enumerate(lengths):
            attention_mask[i, :length] = 1
        return ~attention_mask

    def masked_mean(self, x, lengths, dim=0):  
        batch_size = x.size(0)  
        result = []  
        for i in range(batch_size): 
            if isinstance(lengths, int):
                length = lengths
            else:
                length = lengths[i]  
            sliced = x[i, :length, :]  
            mean_value = sliced.mean(dim=dim)  
            result.append(mean_value)  
        result_tensor = torch.stack(result)  
        return result_tensor 
    
    def forward(self, x, lengths):
        x = self.embedding(x) + self.position_encoding[:, :x.size(1), :]

        attention_mask = self.create_attention_mask(lengths, x.size(1)).to(x.device)

        x = self.transformer(x, src_key_padding_mask = attention_mask)
        x = self.layer_norm(x)  
        x = self.masked_mean(x, lengths, dim=0)
        
        logits = self.fc(x)
        return logits
