from torchvision.datasets import CIFAR100
from dataset import train_transform, valid_transform, create_dataloader
from utils import get_device, ToDeviceLoader, show_batch

if __name__ == '__main__':
    train_transform = train_transform()
    valid_transform = valid_transform()

    train_data = CIFAR100(download=True, root="./data", transform=train_transform)
    valid_data = CIFAR100(root="./data", train=False, transform=valid_transform)

    train_dataloader, valid_dataloader = create_dataloader(train_data, valid_data, batch_size=16)

    device = get_device()
    print(device)

    train_dl = ToDeviceLoader(train_dataloader, device)
    test_dl = ToDeviceLoader(valid_dataloader, device)

    for batch in train_dl:
        images, labels = batch
        print(images.shape)
        print(labels)
        break


    # Train()

    # net = coatnet_1()
    # out = net(img)
    # print(out.shape, count_parameters(net))
    #
    # net = coatnet_2()
    # out = net(img)
    # print(out.shape, count_parameters(net))
    #
    # net = coatnet_3()
    # out = net(img)
    # print(out.shape, count_parameters(net))
    #
    # net = coatnet_4()
    # out = net(img)
    # print(out.shape, count_parameters(net))
