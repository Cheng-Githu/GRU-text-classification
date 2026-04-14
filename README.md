# GRU 文本二分类模型
基于 BERT + GRU 构建的文本二分类模型，通过超参数网格搜索优化模型性能，代码已做随机性固定处理，可稳定复现运行结果。
# 项目简介
本项目实现了以 BERT-base-uncased 作为文本特征提取器、GRU 作为序列特征建模层的文本二分类方案，核心特性包括：
冻结 BERT 权重，仅训练 GRU 和全连接层，降低训练成本；
内置超参数网格搜索，自动筛选最优参数组合；
固定随机种子，保证结果可复现；
适配 GPU/CPU 运行环境，自动检测并切换。
# 环境准备
## 1. 系统依赖
Python 3.8+
CUDA 11.7+（可选，GPU 加速）
## 2. 安装 Python 依赖

pip install -r requirements.txt
## 3. 依赖清单（requirements.txt）
新建 requirements.txt 文件，写入以下内容：
txt
torch>=2.0.0
transformers==4.30.2
scikit-learn==1.2.2
numpy>=1.24.0
pickle-mixin==1.0.2
# 数据集准备
## 1. 数据集格式
需准备预处理后的 news_dataset.pkl 文件（放置在项目根目录），文件为 Pickle 序列化的字典，结构如下：
python
运行
{
    "X_train": [text1, text2, ...],  # 训练集文本列表，str 类型
    "X_test": [text1, text2, ...],   # 测试集文本列表，str 类型
    "y_train": [0, 1, 0, ...],       # 训练集标签，0/1 二分类
    "y_test": [0, 1, 0, ...]         # 测试集标签，0/1 二分类
}
## 2. 数据要求
文本需为英文（适配 BERT-base-uncased 分词器）；
单条文本长度建议不超过 200 字符（代码中 MAX_LEN=200 会自动截断 / 补全）。
# 模型训练
## 1. 运行训练
python train.py
## 2. 运行日志示例
plaintext
 Using device: cuda
Params: {'hidden_dim': 128, 'num_layers': 1, 'lr': 0.001, 'bidirectional': True} | Acc: 0.8925
Params: {'hidden_dim': 256, 'num_layers': 1, 'lr': 0.0005, 'bidirectional': True} | Acc: 0.9158

BEST FINAL RESULT:
Best Accuracy: 0.7922
Best Params: {'hidden_dim': 128, 'num_layers': 1, 'lr': 0.001, 'bidirectional': True}
# 模型结构
层级	细节说明
BERT 层	使用 bert-base-uncased，冻结所有参数，输出 768 维文本特征
GRU 层	可配置隐藏维度 / 层数 / 是否双向，输入 768 维，输出维度 = hidden_dim × (2 if 双向 else 1)
全连接层	将 GRU 输出映射为 1 维概率值
激活函数	Sigmoid，输出 0~1 概率，阈值 0.5 划分二分类标签
#超参数说明
## 1. 待搜索超参数网格（model.py）
def get_param_grid():
    return [
        {"hidden_dim": 128, "num_layers": 1, "lr": 1e-3, "bidirectional": True},
        {"hidden_dim": 256, "num_layers": 1, "lr": 5e-4, "bidirectional": True},
    ]
## 2. 关键参数解释
表格
参数名	说明
hidden_dim	GRU 隐藏层维度，可选 128/256
num_layers	GRU 层数，当前固定为 1（多层可开启 dropout=0.3）
lr	优化器学习率，可选 1e-3/5e-4
bidirectional	是否使用双向 GRU，当前固定为 True


