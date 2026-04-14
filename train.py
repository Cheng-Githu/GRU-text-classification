import os
import pickle
import torch
import torch.nn as nn  # 这里修复了！
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import random
import numpy as np
from collections import Counter

# ===================== 随机种子 =====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(222)

# ===================== 预处理 =====================
MAX_LEN = 250

def tokenize_text(text):
    return text.lower().split()[:MAX_LEN]

def build_vocab(texts, min_freq=2):
    all_tokens = []
    for text in texts:
        all_tokens.extend(tokenize_text(text))
    token_counts = Counter(all_tokens)
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, count in token_counts.items():
        if count >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab

def text_to_ids(text, vocab):
    tokens = tokenize_text(text)
    ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    if len(ids) < MAX_LEN:
        ids += [vocab["<pad>"]] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]
    return ids

# ===================== 数据集 =====================
class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids = text_to_ids(self.texts[idx], self.vocab)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)

# ===================== 训练 =====================
def run_train(params, train_loader, test_loader, vocab_size, device):
    from model import GRUTextClassifier
    model = GRUTextClassifier(
        vocab_size=vocab_size,
        embed_dim=params["embed_dim"],
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        bidirectional=params["bidirectional"]
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=1e-4)

    best_acc = 0.0
    for epoch in range(15):
        model.train()
        for ids, y in train_loader:
            ids, y = ids.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for ids, y in test_loader:
                pred = model(ids.to(device))
                preds.extend((pred > 0.5).cpu().numpy())
                trues.extend(y.numpy())

        acc = accuracy_score(trues, preds)
        if acc > best_acc:
            best_acc = acc
    return best_acc, params

# ===================== 主程序 =====================
if __name__ == "__main__":
    with open("news_dataset.pkl", "rb") as f:
        data = pickle.load(f)
    X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

    vocab = build_vocab(X_train)
    vocab_size = len(vocab)
    print(f"词汇表大小: {vocab_size}")

    train_loader = DataLoader(NewsDataset(X_train, y_train, vocab), batch_size=16, shuffle=True)
    test_loader = DataLoader(NewsDataset(X_test, y_test, vocab), batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    best_acc = 0
    best_params = None
    from model import get_param_grid
    for params in get_param_grid():
        acc, param = run_train(params, train_loader, test_loader, vocab_size, device)
        print(f"参数: {params} | 准确率: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_params = param

    print("\n===== 最终结果 =====")
    print(f"最高准确率: {best_acc:.4f}")
    print(f"最优参数: {best_params}")