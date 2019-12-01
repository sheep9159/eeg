import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import time

BATCH_SIZE = 64
LEARNING_RATE = 0.001

device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
-------------------------------------------------------------
    此处要用在matlab上生成的脑电功率谱地形图改成自己的database
'''

train_data = torchvision.datasets.MNIST(
    'mnist', train=True, transform=torchvision.transforms.ToTensor(), download=False
)
test_data = torchvision.datasets.MNIST(
    'mnist', train=False, transform=torchvision.transforms.ToTensor(), download=False

)
'''
-------------------------------------------------------------    
'''

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            # n = ((in_channels - kernel + 2 * padding) / stride) + 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # n = in_channels / 2
            nn.MaxPool2d(2)
            # batch_size * 32 * 14 * 14
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, groups=32),
            nn.ReLU(),
            nn.MaxPool2d(2)
            # batch_size * 64 * 7 * 7
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, groups=64),
            nn.ReLU(),
            nn.MaxPool2d(2)
            # batch_size * 64 * 3 * 3  maxpool2d整除不了就省去如 7 / 2 = 3.5取 3
        )
        self.linear = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)  # 将平面的(即有形状的矩阵)平展成向量,63 * 3* 3 = 576维的向量
        out = self.linear(res)

        return out


convNet = Net()
convNet = convNet.to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(convNet.parameters(), lr=LEARNING_RATE)

for epoch in range(5):
    for step, (batch_train_data, batch_train_label) in enumerate(train_loader):
        batch_train_label = batch_train_label.to(device)
        batch_train_data = batch_train_data.to(device)
        prediction = convNet(batch_train_data)

        loss = loss_func(prediction, batch_train_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print('epoch', epoch, ' | ', 'step', step, ' | ', 'loss: ', loss)

with torch.no_grad():
    for step, (batch_test_data, batch_test_label) in enumerate(test_loader):
        batch_test_data = batch_test_data.to(device)
        batch_test_label = batch_test_label.to(device)
        pred = convNet(batch_test_data)
        pred = torch.max(pred, 1)[1].cpu().numpy()
        print(pred)
        print(batch_test_label.cpu().numpy())
        break


print('use time: ', time.process_time(), 's')