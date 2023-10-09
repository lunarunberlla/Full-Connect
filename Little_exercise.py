
import torch

'''实现softmax函数'''
def softmax(x):
    e_x = torch.exp(x - torch.max(x))
    return e_x / e_x.sum(dim=1, keepdim=True)




'''实现sigmoid函数'''
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))



'''实现softmax交叉熵损失函数'''
def cross_softmax(y_true,y_pred):
    y_pred = torch.softmax(y_pred, dim=1)
    # 对预测的概率进行稳定的log运算
    log_y_pred = torch.log(y_pred + 1e-7)
    # 创建一个one-hot矩阵
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(1, y_true.long(), 1)
    # 与one-hot标签相乘，然后取负数，然后求和。最后求平均值
    loss = - torch.mean(log_y_pred * one_hot)

    return loss


'''实现sigmoid交叉熵损失函数'''
def cross_sigmoid(y_true,y_pred):
    y_pred_sig = sigmoid(y_pred)
    loss = -y_true * torch.log(y_pred_sig) - (1 - y_true) * torch.log(1 - y_pred_sig)
    return torch.mean(loss)