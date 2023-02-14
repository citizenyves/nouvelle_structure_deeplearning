import torch
import torch.nn as nn
from torch.optim import Adam

from coatnet import coatnet_0

def Train(epochs, trainDL, validDL):

    best_accuracy = 0.0

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    net = coatnet_0()
    net.to(device)

    # Epochs
    num_epochs = 100
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0

        for i, (images, labels) in enumerate(trainDL, 0):

            # Forward
            out = net(img)
            # print(out.shape, count_parameters(net))

            # LossFunction


            # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
            loss_fn = nn.CrossEntropyLoss()
            optimizer = Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
            loss = loss_fn(out, labels)
                # Backward
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()  # extract the loss value
            if i % 1000 == 999:
                # print every 1000 (twice per epoch)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                running_loss = 0.0

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy = testAccuracy()
        print('For epoch', epoch + 1, 'the test accuracy over the whole test set is %d %%' % (accuracy))
