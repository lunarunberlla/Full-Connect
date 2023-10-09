from torchvision import datasets, transforms
import torch
import torch.nn as nn
from torch import optim
import os
import random

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Checking if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataLoader:

    def load_mnist(self, batch_size=64):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.3,), (0.3,)),
            transforms.Lambda(lambda x: torch.cat((x.view(-1), torch.ones(1, device=x.device))))
        ])
        trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
        testset = datasets.MNIST('mnist_test', train=False, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

        return trainloader, testloader


class Model(nn.Module):
    def __init__(self, ndim, class_num=10):
        super(Model, self).__init__()
        # self.w1 = nn.Parameter(torch.rand(ndim, 785).to(device) * 0.2 - 0.1)
        # self.w2 = nn.Parameter(torch.rand(785, class_num).to(device) * 0.2 - 0.1)
        self.l1=nn.Linear(ndim,785)
        self.f=nn.Sigmoid()
        self.l2=nn.Linear(785,10)
    def forward(self, x):

        # self.A=torch.softmax(x@self.w1,dim=1)
        # return self.A@self.w2
        self.A = self.l1(x)
        self.B = self.f(self.A)
        return self.l2(self.B)


class Params:
    def __init__(self, lamuda=0.01, batch_size=64, class_num=10,feature_num=785,epoch=20):
        self.lamuda = lamuda
        self.batch_size = batch_size
        self.class_num = class_num
        self.feature_num=feature_num
        self.epoch=epoch


class Criterion:
    def cross_entropy_loss(self, y_true, y_pred):
        y_pred = torch.softmax(y_pred, dim=1)
        log_y_pred = torch.log(y_pred + 1e-7)
        one_hot = torch.zeros_like(y_pred)
        one_hot.scatter_(1, y_true.long().unsqueeze(1), 1)
        loss = - torch.mean(log_y_pred * one_hot)

        return loss


class Utils:
    def __init__(self, params):

        self.params = params
        self.trainloader, self.testloader = DataLoader().load_mnist(self.params.batch_size)
        self.model = Model(ndim=self.params.feature_num, class_num=self.params.class_num).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)


    def train_one_step(self, xs, ys):
        xs, ys = xs.to(device), ys.to(device)
        self.optimizer.zero_grad()
        y_preds = self.model(xs)
        loss = Criterion().cross_entropy_loss(ys, y_preds)
        loss.backward()
        self.optimizer.step()
        return loss

    def predict(self, xs):
        xs = xs.to(device)
        with torch.no_grad():
            y_preds = self.model(xs)
        return y_preds

    def evaluate(self, ys, ys_pred):
        ys, ys_pred = ys.to(device), ys_pred.to(device)
        ys_pred = torch.argmax(ys_pred, dim=1)
        diff_count = (len(ys) - torch.sum(torch.logical_not(torch.eq(ys, ys_pred)))).float()
        return diff_count / len(ys)


class Ways(Utils):
    def __init__(self, params):

        super(Ways, self).__init__(params)


    def Softmax(self):

        for epoch in range(self.params.epoch):
            for x, y in self.trainloader:
                x, y = x.to(device), y.to(device)
                loss = self.train_one_step(x,y)
            if epoch%1==0:
                print(f'loss at epoch {epoch} is {loss.item():.4f}')

        train_accuracy = self.evaluate_batch(self.trainloader)
        test_accuracy = self.evaluate_batch(self.testloader)
        print(f'Softmax->Training set: precision= {train_accuracy}')
        print(f'Softmax->Test set: precision={test_accuracy}')

    def evaluate_batch(self, dataloader):
        total_accuracy = 0
        total_count = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
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

    params = Params(lamuda=0.01,batch_size=2048,feature_num=785,class_num=10,epoch=30)
    process = RunProcess(params)
    process.run()
    # out
    # Softmax->Training set: precision= 0.9985333333333334
    # Softmax->Test set: precision=0.9754000044822693