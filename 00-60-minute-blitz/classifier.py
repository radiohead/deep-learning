import torchvision
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt


class CIFAR10Classifier(nn.Module):
    def __init__(self):
        super(CIFAR10Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def predict(self, input):
        output = self(input)
        _, predicted = torch.max(output.data, 1)

        return predicted


class CIFAR10Data(object):
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.trainset = torchvision.datasets.CIFAR10(
            root='./data', download=True, train=True, transform=self.transform)
        self.testset = torchvision.datasets.CIFAR10(
            root='./data', download=True, train=False, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=4, shuffle=True, num_workers=2)
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=4, shuffle=False, num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')


def imgshow(img: torch.Tensor):
    img = img / 2 + 0.5
    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    traindata = CIFAR10Data()
    net = CIFAR10Classifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print('Traning starting...')

    for epoch in range(10):
        epoch_loss = 0.0

        for i, data in enumerate(traindata.trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, epoch_loss / 2000))
                epoch_loss = 0.0

    print('Training done')
    path = './cifar_net.pth'
    torch.save(net.state_dict(), path)
    print('Saved model at %s' % path)
