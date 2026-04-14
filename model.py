import torch
import torch.nn as nn


class GRUTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, num_layers=2, bidirectional=True, dropout=0.5):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        self.layer_norm = nn.LayerNorm(hidden_dim * 2 if bidirectional else hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids):
        embed_out = self.embedding(input_ids)
        gru_out, _ = self.gru(embed_out)

        # 平均池化 + 归一化
        avg_pool = torch.mean(gru_out, dim=1)
        feat = self.layer_norm(avg_pool)
        feat = self.dropout(feat)

        out = self.fc(feat)
        return self.sigmoid(out).squeeze()


def get_param_grid():
    return [
        {"embed_dim": 256, "hidden_dim": 256, "num_layers": 2, "bidirectional": True, "lr": 1e-3},
        {"embed_dim": 256, "hidden_dim": 256, "num_layers": 2, "bidirectional": True, "lr": 5e-4},
    ]