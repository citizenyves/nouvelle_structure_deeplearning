from tqdm import tqdm
import torch
from torch import flatten
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchvision.datasets import CIFAR100
from dataset import train_transform, valid_transform, create_dataloader
from utils import get_device, ToDeviceLoader, show_batch


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # première
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.convbn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=4, stride=1)
        self.convbn2 = nn.BatchNorm2d(12)

        self.fc1 = nn.Linear(12 * 5 * 5, 120)
        self.bn1 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn2 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 100) # prédiction

        # deuxième
        self.fc4 = nn.Linear(100, 12 * 5 * 5)
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=28, kernel_size=3, stride=1)
        # self.conv3 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, stride=1)
        self.convbn3 = nn.BatchNorm2d(28)
        self.fc5 = nn.Linear(28 * 3 * 3, 84)
        self.bn5 = nn.BatchNorm1d(84)
        self.fc6 = nn.Linear(84, 100) # prédiction

    def forward(self, x):
        x = self.pool(F.relu(self.convbn1(self.conv1(x))))
        x_stack1 = self.pool(F.relu(self.convbn2(self.conv2(x))))
        x = torch.flatten(x_stack1, 1) # flatten all dimensions except batch
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x, x_stack1

    def forward2(self, x, x_stack1):
        x = F.relu(self.fc4(x)) # 12 * 5 * 5
        x_stack2 = torch.reshape(x, (16, 12, 5, 5)) # (16, 12, 5, 5)
        x = torch.concat((x_stack1, x_stack2), 1) # (16, 12, 5, 5) concat (16, 12, 5, 5) -> (16, 24, 5, 5)
        x = F.relu(self.convbn3(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.bn5(F.relu(self.fc5(x)))
        x = self.fc6(x)
        return x

# class Net2(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, stride=1)
#         self.convbn1 = nn.BatchNorm2d(16)
#
#         self.fc1 = nn.Linear(16 * 3 * 3, 84)
#         self.bn1 = nn.BatchNorm1d(84)
#         self.fc2 = nn.Linear(84, 100)
#
#     def forward(self, x):
#         x = F.relu(self.convbn3(self.conv3(x)))  # torch.Size([16, 16, 3, 3])
#         x = torch.flatten(x, 1)  # flatten all dimensions except batch
#         x = self.bn1(F.relu(self.fc1(x)))
#         x = self.fc2(x)
#         return x

def train(net, train_dataloader, datainfo, rate=2):
    for epoch in range(datainfo[1]):  # loop over the dataset multiple times

        running_loss = 0.0
        correct = 0
        total = 0

        # defining the loss function and the optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        # checking if GPU is available
        if torch.cuda.is_available():
            net = net.cuda()
            criterion = criterion.cuda()

        # training
        for batch_i, (inputs, labels) in enumerate(tqdm(train_dataloader)):

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, x_stack1 = net.forward(inputs)

            # if (batch_i != 0) and (batch_i % rate == 0):
            #     outputs = net.forward2(outputs, x_stack1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1) # predicted labels (1, 16)
            total += labels.size(0) # 16 = batch_size
            correct += predicted.eq(labels.data).cpu().sum()

        # if batch_i % 125 == 124:    # print every 2000 mini-batches
        print('| Epoch [%3d/%3d] \t\tLoss: %.4f Acc: %.3f%%'
                     % (epoch+1, epochs, loss.item(), correct / total * 100.))
        # print(f'[{epoch + 1}, {batch_i + 1:5d}] loss: {running_loss / 2000:.3f}')
        # running_loss = 0.0

            # # batch 저장, acc 저장
            # if epoch >= 4: # à partir de 5em Epoch
            #     outputs2 = net2(outputs)


    print('Finished Training')
    PATH = './weights/convnet.pth'
    torch.save(net.state_dict(), PATH)

def test(net, valid_dataloader):
    correct = 0
    total = 0
    # 학습 중이 아니므로, 출력에 대한 변화도를 계산할 필요가 없습니다
    with torch.no_grad():
        for data in valid_dataloader:
            images, labels = data
            # 신경망에 이미지를 통과시켜 출력을 계산합니다
            outputs = net(images)
            # 가장 높은 값(energy)를 갖는 분류(class)를 정답으로 선택하겠습니다
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


if __name__ == '__main__':

    test = True

    # import dataset
    train_transform = train_transform()
    valid_transform = valid_transform()

    # tr = 50000, vl = 10000
    train_data = CIFAR100(download=False, root="/Users/taeyeon/Projet/data", transform=train_transform)
    valid_data = CIFAR100(root="/Users/taeyeon/Projet/data", train=False, transform=valid_transform)

    # data informations
    lendata = len(train_data)
    epochs = 50
    batch_size = 512
    datainfo = (lendata, epochs, batch_size)

    # Create dataloader
    train_dataloader, valid_dataloader = create_dataloader(train_data, valid_data, batch_size=batch_size)

    # device checking
    device = get_device()
    print(device)
    train_dataloader = ToDeviceLoader(train_dataloader, device)
    valid_dataloader = ToDeviceLoader(valid_dataloader, device)


    # defining the model
    net = Net()
    net.to(device)
    print(net)
    # net2 = Net2()
    # print(net, net2)

    train(net, train_dataloader, datainfo, rate=4)

    # test
    del net
    net = Net()
    if test:
        PATH = './weights/convnet.pth'
        net.load_state_dict(torch.load(PATH))
        net.to(device)
    print("- - - test ! - - -")
    test(net, valid_dataloader)




