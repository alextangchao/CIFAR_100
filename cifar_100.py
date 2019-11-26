# -*- coding: utf-8 -*-

import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from resnet import ResNet18


def showpicture():
    # @title show the picture
    import matplotlib.pyplot as plt
    import numpy as np

    # functions to show an image

    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    # print(images)
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


def print_time(time):
    hour = time // 3600
    time %= 3600
    minute = time // 60
    time %= 60
    print("Training time: ", end="")
    if hour > 0:
        print(str(int(hour)) + " hours ", end="")
    print(str(int(minute)) + " minutes " + str(int(time)) + " seconds")


# Training the model
def train(text):
    print("Start training...")
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    n = len(trainloader)
    fout = open(".\\graph\\data.txt", "w")
    fout.write(str(num_epochs) + " " + text + "\n")
    best_accuracy = 0
    start = time.time()
    for epoch in range(num_epochs):
        # net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # inputs = Variable(inputs).cuda()
            # labels = Variable(labels).cuda()
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        print('[%d, %d] loss: %.3f' % (epoch + 1, num_epochs, running_loss / n))
        train_correctness = check_correctness(trainloader)
        test_correctness = check_correctness(testloader)
        best_accuracy = max(test_correctness, best_accuracy)
        fout.write(str(running_loss / n) + " " + str(train_correctness) + " " + str(test_correctness) + "\n")

    end = time.time()
    fout.close()
    print("Finished Training")
    print_time(end - start)
    print("Best accuracy:", best_accuracy)


def check_correctness(loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            # images = Variable(images).cuda()
            # labels = Variable(labels).cuda()
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    correctness = 100 * correct / total

    word = ""
    if loader == trainloader:
        word = "train"
    elif loader == testloader:
        word = "test"
    print('Accuracy of the network on the ' + word + ' images: %.2f %%' % correctness)
    return correctness


def save_model():
    torch.save(net.state_dict(),
               "D:\study\CIFAR_100\model\\" + str(test_correctness) + "_" + str(train_correctness) + "_" + str(
                   int(time.time())) + ".pkl")


def load_model(dir):
    net = torch.load(dir)
    print("Finish loading the data")
    # check_correctness(trainloader)
    check_correctness(testloader)
    exit()


# Define model class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channel, 6 output channels
        # 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 100)
        # self.fc3 = nn.Linear(600, 400)
        # self.fc4 = nn.Linear(400, 200)
        # self.fc5 = nn.Linear(200, 100)
        self.relu = nn.ELU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 1024)
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # x = self.relu(self.fc3(x))
        # x = self.relu(self.fc4(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    num_epochs = 100  # number of times which the entire dataset is passed throughout the model
    batch_size = 128  # the size of input data took for one iteration
    LR = 0.001

    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
         transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
         transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                             download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
    print(device)

    # showpicture()
    # exit()

    net = ResNet18()
    net = net.to(device)

    # load_model("D:\study\CIFAR_100\model\\25.67_76.534_1572670736.pkl")

    train("train test")

    train_correctness = check_correctness(trainloader)
    test_correctness = check_correctness(testloader)
    save_model()
