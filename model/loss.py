import torch.nn.functional as F
import torch


def ctc_loss(output, target):

    S = 8  # 目标序列的长度
    S_min = 5

    output = output.squeeze(dim=2)
    output = output.permute(2,0,1)  # 对应为序列长度，批大小和类别数目

    N = output.shape[1]  # 这里是Batch size
    T = output.shape[0]  # 输入序列的长度，这里为20，所有样本都是等长

    log_probs = output.log_softmax(2).requires_grad_()
    input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
    target_lengths = torch.full(size=(N, ), fill_value=S_min, dtype=torch.long)
    return F.ctc_loss(log_probs,  target, input_lengths, target_lengths, blank=20)

def aug_loss(output, target):

    S = 8 # 目标序列的长度
    S_min = 5
    alpha = 0.2

    output = output.squeeze(dim=2)
    output = output.permute(2,0,1)  # 对应为序列长度，批大小和类别数目

    N = output.shape[1]  # 这里是Batch size
    T = output.shape[0]  # 输入序列的长度，这里为20，所有样本都是等长

    log_probs = output.log_softmax(2).requires_grad_()

    # 求y~=y*%10
    target_ = target % 10

    input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
    target_lengths = torch.full(size=(N, ), fill_value=S_min, dtype=torch.long)
    return F.ctc_loss(log_probs,  target, input_lengths, target_lengths, blank=20) + alpha * F.ctc_loss(log_probs,  target_, input_lengths, target_lengths, blank=20)
