import torch
import torch.nn as nn
from transformers import BertModel

class GRUTextClassifier(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=1, bidirectional=True):
        super().__init__()

        # BERT
        self.bert = BertModel.from_pretrained("bert-base-uncased",
    local_files_only=False,  # 优先从本地加载，无则下载
    cache_dir="./bert_cache"  # 缓存BERT模型到本地，避免重复下载
)
        for param in self.bert.parameters():
            param.requires_grad = False

        # GRU
        self.gru = nn.GRU(
            input_size=768,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.3 if num_layers > 1 else 0
        )

        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(out_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        gru_out, _ = self.gru(bert_out)
        feat = gru_out[:, -1, :]
        out = self.fc(feat)
        return self.sigmoid(out).squeeze()

def get_param_grid():
    return [
        {"hidden_dim": 128, "num_layers": 1, "lr": 1e-3, "bidirectional": True},
        {"hidden_dim": 256, "num_layers": 1, "lr": 5e-4, "bidirectional": True},
    ]