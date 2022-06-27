import os
import torch
from torch.nn import functional as F
from torch import nn
from DataLoader import load_data
from utils import try_gpu, Accumulator, Timer, accuracy, evaluate_accuracy_gpu

class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()

        self.p1_1 = nn.Conv2d(1, 1, kernel_size=(3, 100))
        self.p1_2 = nn.AdaptiveAvgPool2d((3, 1))

        self.p2_1 = nn.Conv2d(1, 1, kernel_size=(5, 100))
        self.p2_2 = nn.AdaptiveAvgPool2d((3, 1))
        
        self.p3_1 = nn.Conv2d(1, 1, kernel_size=(7, 100))
        self.p3_2 = nn.AdaptiveAvgPool2d((3, 1))

    def forward(self, x):
        p1 = self.p1_2(F.relu(self.p1_1(x)))
        p2 = self.p2_2(F.relu(self.p2_1(x)))
        p3 = self.p3_2(F.relu(self.p3_1(x)))
        # print(torch.cat((p1, p2, p3), dim=1).shape)
        return torch.cat((p1, p2, p3), dim=1)

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

net = nn.Sequential(
    Inception(),
    nn.Flatten(),
    nn.Linear(9, 2),
)

batch_size = 512
lr = 0.005
num_epochs = 10

train_iter = load_data(batch_size)
device = try_gpu()
net.apply(init_weights)
print('training on', device)
net.to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()
timer, num_batches = Timer(), len(train_iter)
for epoch in range(num_epochs):
    metric = Accumulator(3)
    net.train()
    for i, (X, y) in enumerate(train_iter):
        timer.start()
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        X = X.unsqueeze(1)
        # print(X.shape)
        y_hat = net(X)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
        with torch.no_grad():
            metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
        timer.stop()
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
    # test_acc = evaluate_accuracy_gpu(net, test_iter)
print(f'loss {train_l:.3f}, train acc {train_acc:.3f}')
print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec 'f'on {str(device)}')
