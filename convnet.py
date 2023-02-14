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
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=4, stride=1)
        self.fc1 = nn.Linear(12 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(net, train_dataloader, datainfo):
    for epoch in range(datainfo[1]):  # loop over the dataset multiple times

        running_loss = 0.0
        correct = 0
        total = 0

        # defining the loss function and the optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.0001)

        # checking if GPU is available
        if torch.cuda.is_available():
            net = net.cuda()
            criterion = criterion.cuda()

        # training
        for batch_i, (inputs, labels) in enumerate(train_dataloader):

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1) # predicted labels (1, 16)
            total += labels.size(0) # 16 = batch_size
            correct += predicted.eq(labels.data).cpu().sum()

            if batch_i % 125 == 124:    # print every 2000 mini-batches
                print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                             % (epoch+1, epochs, batch_i+1, (datainfo[0] // datainfo[2]), loss.item(), correct / total * 100.))
                # print(f'[{epoch + 1}, {batch_i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    PATH = '.weights/convnet.pth'
    torch.save(net.state_dict(), PATH)

if __name__ == '__main__':

    # import dataset
    train_transform = train_transform()
    valid_transform = valid_transform()

    # tr = 50000, vl = 10000

    train_data = CIFAR100(download=False, root="/Users/taeyeon/Projet/data", transform=train_transform)
    valid_data = CIFAR100(root="/Users/taeyeon/Projet/data", train=False, transform=valid_transform)

    # data informations
    lendata = len(train_data)
    epochs = 100
    batch_size = 16
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
    print(net)

    train(net, train_dataloader, datainfo)


