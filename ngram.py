import torch.nn as nn
import torch.nn.functional as F


class NGramLanguageModeler(nn.Module):

    # 初始化时需要指定：单词表大小、想要嵌入的维度大小、上下文的长度
    def __init__(self, vocab_size, embedding_dim, context_size):
        # 继承自nn.Module，例行执行父类super 初始化方法
        super(NGramLanguageModeler, self).__init__()
        # 建立词嵌入模块
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 线性层1
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        # 线性层2，隐藏层 hidden_size 为128
        self.linear2 = nn.Linear(128, vocab_size)

    # 重写的网络正向传播方法
    # 只要正确定义了正向传播
    # PyTorch 可以自动进行反向传播
    def forward(self, inputs):
        # 将输入进行“嵌入”，并转化为“行向量”
        embeds = self.embeddings(inputs).view((1, -1))
        # 嵌入后的数据通过线性层1后，进行非线性函数 ReLU 的运算
        out = F.relu(self.linear1(embeds))
        # 通过线性层2后
        out = self.linear2(out)
        # 通过 log_softmax 方法将结果映射为概率的log
        # log 概率是用于下面计算负对数似然损失函数时方便使用的
        log_probs = F.log_softmax(out)
        return log_probs