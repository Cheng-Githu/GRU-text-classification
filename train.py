import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import pickle
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer
import random
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(222)
# ===================== BERT 分词器（修复路径错误）=====================
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased",
    local_files_only=False
)
MAX_LEN = 200

# ===================== 数据集 =====================
class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].squeeze()
        mask = inputs["attention_mask"].squeeze()

        return input_ids, mask, torch.tensor(label, dtype=torch.float32)

# ===================== 训练 =====================
def run_train(params, train_loader, test_loader, device):
    from model import GRUTextClassifier
    model = GRUTextClassifier(
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        bidirectional=params["bidirectional"]
    ).to(device)

    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])

    best_acc = 0.0
    for epoch in range(10):
        model.train()
        for ids, mask, y in train_loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids, mask), y)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for ids, mask, y in test_loader:
                ids, mask = ids.to(device), mask.to(device)
                pred = model(ids, mask)
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
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    train_loader = DataLoader(NewsDataset(X_train, y_train), batch_size=8, shuffle=True)
    test_loader = DataLoader(NewsDataset(X_test, y_test), batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(" Using device:", device)

    best_acc = 0
    best_params = None
    from model import get_param_grid
    for params in get_param_grid():
        acc, param = run_train(params, train_loader, test_loader, device)
        print(f"Params: {params} | Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_params = params

    print("\n BEST FINAL RESULT:")
    print(f"Best Accuracy: {best_acc:.4f}")
    print(f"Best Params: {best_params}")