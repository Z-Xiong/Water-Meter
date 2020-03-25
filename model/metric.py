import torch
from .decode import feature_to_y

def accuracy(output, target):
    output = feature_to_y(output)   #  把特征向量解码为标签序列
    if torch.cuda.is_available():
        target = target.cpu()
    target = target.numpy().tolist()  # 把目标序列从tensor转换为列表
    with torch.no_grad():
        correct=0
        for i in range(len(output)): # 计算序列相同的个数
            if output[i]==target[i]:
                correct =correct+1
    return correct / len(output)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)