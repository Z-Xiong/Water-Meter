import torch
import torch.nn.functional as F

C = {0: 0,
     1: 1,
     2: 2,
     3: 3,
     4: 4,
     5: 5,
     6: 6,
     7: 7,
     8: 8,
     9: 9,
     10: 10,
     11: 11,
     12: 12,
     13: 13,
     14: 14,
     15: 15,
     16: 16,
     17: 17,
     18: 18,
     19: 19,
     20: 21}
# 定义函数delet_char, 去重序列，并删除空白字符b
def get_preTarget(a):
    O=[]
    print(a)
    for i in range(len(a)):
        o = list(set(a[i]))
        o.sort(key=a[i].index)
        if 'b' in o:
            o.remove(21)
        O.append(o)
    
    return O

# 将得到的预测序列转换为实际读数
def middle_char(y):
    for i in range(len(y)):
        if y[i]>9 and y[i]!=21:
            if i==len(y):
                y[i]=y[i]-9.5
            else:
                y[i] = y[i]-10
    return y

def feature_to_y(x):
    x = F.softmax(x, dim=1) # 把特征序列转换为概率
    x, index = torch.max(x, dim=1) # 选出概率最大的那个
    index = torch.squeeze(index) # 压缩index维度
    index = index.numpy().tolist() # 把index转换为list，方便遍历

    b = []
    for i in range(len(index)):
        a=[]
        for j in index[i]:
            a.append(C[j])    # 把概率转换为字符序列
        b.append(a)

    y = get_preTarget(b)    # 字符序列去重，保留原有顺序,并删除空白字符b,这时得到的结果与target一致
    
    return y