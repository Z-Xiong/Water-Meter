import torch.nn.functional as F
import torch


def ctc_loss(output, target):
    T = 20 # 输入序列的长度，这里为20
    C = 21 # 类别的种类数
    N = 16 # 这里是Batch size
    S = 8 # 目标序列的长度
    S_min = 5

    output = output.squeeze(dim=2)
    output = output.permute(2,0,1)

    log_probs = output.log_softmax(2).requires_grad_()
    input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
    target_lengths = torch.full(size=(N, ), fill_value=S_min, dtype=torch.long)
    return F.ctc_loss(log_probs,  target, input_lengths, target_lengths)