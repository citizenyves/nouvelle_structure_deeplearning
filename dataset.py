import torchvision.transforms as Transform
from torch.utils.data.dataloader import DataLoader

def train_transform():
    stats = ((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025))

    train_transform = Transform.Compose([
        Transform.RandomHorizontalFlip(),
        Transform.RandomCrop(32, padding=4, padding_mode="reflect"),
        Transform.ToTensor(),
        Transform.Normalize(*stats)
    ])
    return train_transform

def valid_transform():
    stats = ((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025))

    valid_transform = Transform.Compose([
        Transform.ToTensor(),
        Transform.Normalize(*stats)
    ])
    return valid_transform

# def CIFAR100(exist=True):
#     if exist:
#         train_data = CIFAR100(root="./data", transform=train_transform)
#     else:
#         train_data = CIFAR100(download=True, root="./data", transform=train_transform)
#     valid_data = CIFAR100(root="./data", train=False, transform=valid_transform)
#
#     return train_data, valid_data

def create_dataloader(train_data, valid_data, batch_size, num_workers=4, pin_memory=True):
    train_dataloader = DataLoader(train_data, batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size, num_workers=num_workers, pin_memory=pin_memory)
    return train_dataloader, valid_dataloader
