from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class DataLoader:

    def load_mnist(self, batch_size=64):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.3,), (0.3,)),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
        testset = datasets.MNIST('mnist_test', train=False, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
        return trainloader, testloader


class Model(nn.Module):
    def __init__(self, ndim, class_num=10):
        super(Model, self).__init__()
        #self.fc = nn.Linear(ndim, class_num)
        self.w = nn.Parameter(torch.rand(ndim, class_num) * 0.2 - 0.1)

    def forward(self, x):
        #return self.fc(x)
        return x @ self.w


class Params:
    def __init__(self, lamuda=0.01, batch_size=64, class_num=10):
        self.lamuda = lamuda
        self.batch_size = batch_size
        self.class_num = class_num

class Criterion:

    def cross_entropy_loss(self,y_true, y_pred):
        y_pred=torch.softmax(y_pred,dim=1)
        # 对预测的概率进行稳定的log运算
        log_y_pred = torch.log(y_pred + 1e-7)
        # 创建一个one-hot矩阵
        one_hot = torch.zeros_like(y_pred)
        one_hot.scatter_(1, y_true.long().unsqueeze(1), 1)
        # 与one-hot标签相乘，然后取负数，然后求和。最后求平均值
        loss = - torch.mean(log_y_pred * one_hot)

        return loss

class Utils:
    def __init__(self, params):
        self.params = params
        self.trainloader, self.testloader = DataLoader().load_mnist(self.params.batch_size)
        self.model = Model(ndim=784, class_num=self.params.class_num)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def train_one_step(self, xs, ys):
        self.optimizer.zero_grad()
        y_preds = self.model(xs)
        loss = Criterion().cross_entropy_loss(ys, y_preds)
        loss.backward()
        self.optimizer.step()
        return loss

    def predict(self, xs):
        with torch.no_grad():
            y_preds = self.model(xs)
        return y_preds

    def evaluate(self, ys, ys_pred):
        ys_pred = torch.argmax(ys_pred, dim=1)
        diff_count = (len(ys) -torch.sum(torch.logical_not(torch.eq(ys, ys_pred)))).float()
        return diff_count / len(ys)

class Ways(Utils):
    def __init__(self, params):
        super(Ways, self).__init__(params)

    def Softmax(self):
        for epoch in range(5):
            for x, y in self.trainloader:
                loss = self.train_one_step(x, y)
            if epoch%5==0:
                print(f'loss at epoch {epoch} is {loss.item():.4f}')

        train_accuracy = self.evaluate_batch(self.trainloader)
        test_accuracy = self.evaluate_batch(self.testloader)
        print(f'Softmax->Training set: precision= {train_accuracy}')
        print(f'Softmax->Test set: precision={test_accuracy}')

    def evaluate_batch(self, dataloader):
        total_accuracy = 0
        total_count = 0
        for x, y in dataloader:
            y_pred = self.predict(x)
            total_accuracy += self.evaluate(y, y_pred).item() * len(y)
            total_count += len(y)
        return total_accuracy / total_count

class RunProcess:
    def __init__(self, params):
        self.comp = Ways(params)

    def run(self):
        self.comp.Softmax()

if __name__ == '__main__':
    params = Params(lamuda=0.01)
    process = RunProcess(params)
    process.run()