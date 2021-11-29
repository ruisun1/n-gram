
from  data import word_to_ix,ix_to_word,trigrams,vocab
from  ngram import NGramLanguageModeler
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
# 即 前两个词
CONTEXT_SIZE = 2
# 嵌入维度
EMBEDDING_DIM = 10
losses = []
# 损失函数为 负对数似然损失函数(Negative Log Likelihood)
loss_function = nn.NLLLoss()
# 实例化我们的模型，传入：
# 单词表的大小、嵌入维度、上下文长度
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
# 优化函数使用随机梯度下降算法，学习率设置为0.001
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(1000):
    total_loss = 0
    # 循环context上下文，比如：['When', 'forty']
    # target，比如：winters
    for context, target in trigrams:

        # 步骤1：准备数据
        # 将context如“['When', 'forty']”
        # 转化为索引，如[68, 15]
        # 再建立为 PyTorch Variable 变量，以计算梯度
        context_idxs = list(map(lambda w: word_to_ix[w], context))
        context_var = autograd.Variable( torch.LongTensor(context_idxs) )

        # 步骤2：清空梯度值，防止上次的梯度累计
        model.zero_grad()

        # 步骤3：运行网络的正向传播，获得 log 概率
        log_probs = model(context_var)

        # 步骤4：计算损失函数
        # 传入 target Variable
        loss = loss_function(log_probs, autograd.Variable(torch.LongTensor([word_to_ix[target]])))

        # 步骤5：进行反向传播并更新梯度
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    losses.append(total_loss)

print('Finished')

def evaluate():

    test=['were','to']
    ids=list(map(lambda w:word_to_ix[w],test))
    ids_tensor=torch.LongTensor(ids)
    with torch.no_grad():
        log_probs=model(ids_tensor)
    print(ix_to_word[torch.Tensor.argmax(log_probs).item()])


evaluate()