import torch
import torchvision
from torch import nn as nn
from torch.utils import data
from torchvision import transforms
import os

num_epochs = 10
lr = 0.1
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#if you use the original image size of cifar-10 and custom network, please use this sentence.
trans = transforms.ToTensor()

#if you decide to use original network, you need to enlarge image size to 227*227.
"""trans = []
trans.append(torchvision.transforms.Resize(size=227))
trans.append(torchvision.transforms.ToTensor())
trans = torchvision.transforms.Compose(trans)
"""

cifar10_train = torchvision.datasets.CIFAR10(root='Cifar-10', train=True, download=True, transform=trans)
cifar10_test = torchvision.datasets.CIFAR10(root='Cifar-10', train=False, download=True, transform=trans)
train_iter = data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
test_iter = data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=False)

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter, device=None):
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    acc_sum = 0.0
    n = 0
    for X, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        acc_sum += accuracy(net(X), y)
        n += X.shape[0]
    return float(acc_sum / n)


def train_model(net, train_iter, test_iter, num_epochs, lr, device, net_name):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)

    print("training on", device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    num = 0
    loss_sum = 0.0
    acc_sum = 0.0
    iter = 0

    for epoch in range(num_epochs):
        net.train()
        for i, (X, y) in enumerate(train_iter):
            iter += 1
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                loss_sum += l * X.shape[0]
                acc_sum += accuracy(y_hat, y)
                num += X.shape[0]
        train_l = loss_sum / num
        train_acc = acc_sum / num
        test_acc = evaluate_accuracy(net, test_iter)
        print(f"epoch:{epoch + 1}, loss:{train_l:.3f}, train acc:{train_acc:.3f}, test acc:{test_acc:.3f}")

    #save your model and parameters
    if not os.path.exists('logs/'):
        os.mkdir('logs/')
    torch.save(net, f'logs/{net_name}.pkl')
    torch.save(net.state_dict(), f'logs/{net_name}.pth')

#you can define your own network here, I will give some examples.
def get_alexnet():
    alexnet = nn.Sequential(
        # input 3*32*32
        nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1), nn.ReLU(),  # output 8*32*32
        nn.MaxPool2d(kernel_size=2, stride=2),  # output 8*16*16
        nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),  # output 16*16*16
        nn.MaxPool2d(kernel_size=2, stride=2),  # output 16*8*8
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),  # output 32*8*8
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),  # output 64*8*8
        nn.Conv2d(64, 48, kernel_size=3, stride=1, padding=1), nn.ReLU(),  # output 48*8*8
        nn.MaxPool2d(kernel_size=2, stride=2),  # output 48*4*4
        nn.Flatten(),
        nn.Linear(48 * 4 * 4, 512), nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, 128), nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(128, 10)
    )
    return alexnet

class mobilenet_block(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(mobilenet_block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride,
             padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1,
            stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = nn.functional.relu(self.bn2(self.conv2(out)))
        return out

def get_mobilenet():
    mobilenet = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),  # output16*16*16
        mobilenet_block(16, 16),  # output16*16*16
        nn.Conv2d(16, 32, kernel_size=1), nn.BatchNorm2d(32), nn.ReLU(),  # output16*16*32
        mobilenet_block(32, 32, 2),  # output8*8*32
        nn.Conv2d(32, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),  # output8*8*64
        mobilenet_block(64, 64, 1),  # output8*8*64
        nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),  # output8*8*64
        mobilenet_block(64, 128, 2),  # output4*4*128
        nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(),  # output4*4*128
        mobilenet_block(128, 128, 1),  # output4*4*128
        nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(),  # 输出4*4*128
        mobilenet_block(128, 256, 2),  # 输出2*2*256
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(256, 10)
    )
    return mobilenet


if __name__ == '__main__':
    #the first variate is your network, the last variate is your model name that you will save on your disk
    train_model(get_mobilenet(), train_iter, test_iter, num_epochs, lr, device, 'mobilenet')