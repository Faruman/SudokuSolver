import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid

#neural net imports
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

#import external libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os
import math

#check for cuda
print(torch.cuda.is_available())
print(torch.backends.cudnn.enabled)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print("cpu")

#load data
train_images = np.load("D:/Programming/Python/SudokuSolver/data/moddedMNIST/x_train_ext_invt.npy")
train_labels = np.load("D:/Programming/Python/SudokuSolver/data/moddedMNIST/y_train_ext_invt.npy")
test_images = np.load("D:/Programming/Python/SudokuSolver/data/moddedMNIST/x_test_ext.npy")
test_labels = np.load("D:/Programming/Python/SudokuSolver/data/moddedMNIST/y_test_ext.npy")
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_labels = train_labels.astype('int32')
test_labels = test_labels.astype('int32')

#Training and Validation Split
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, stratify=train_labels, test_size=0.20)

#Convert to tensor and normalize
#train
train_images_tensor = torch.tensor(train_images)/255
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
train_tensor = TensorDataset(train_images_tensor, train_labels_tensor)

#val
val_images_tensor = torch.tensor(val_images)/255
val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)
val_tensor = TensorDataset(val_images_tensor, val_labels_tensor)

#test
test_images_tensor = torch.tensor(test_images)/255

#generate dataloader
train_loader = DataLoader(train_tensor, batch_size=512, num_workers=0, shuffle=True)
val_loader = DataLoader(val_tensor, batch_size=512, num_workers=0, shuffle=True)
test_loader = DataLoader(test_images_tensor, batch_size=512, num_workers=0, shuffle=False)

#network
class Complex_Net(nn.Module):
    def __init__(self):
        super(Complex_Net, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.linear_block = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 25 * 25, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.linear_block(x)
        return x

class Complex2_Net(nn.Module):
    def __init__(self):
        super(Complex2_Net, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        self.linear_block = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(128 * 13 * 13, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.linear_block(x)
        return x

class Complex3_Net(nn.Module):
    def __init__(self):
        super(Complex3_Net, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        self.linear_block = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(64 * 13 * 13, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.linear_block(x)
        return x

class Simple_Net(nn.Module):
    def __init__(self):
        super(Simple_Net, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(p=0.25),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25)
        )

        self.linear_block = nn.Sequential(
            nn.Linear(64 * 13 * 13, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.linear_block(x)
        return x

#initialize model
conv_model = Complex2_Net()
#conv_model = Simple_Net()

#setup optimizer
optimizer = optim.Adam(params=conv_model.parameters(), lr=0.00001)
criterion = nn.CrossEntropyLoss()

exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

#reset the cache
torch.cuda.empty_cache()

if torch.cuda.is_available():
    conv_model = conv_model.cuda()
    criterion = criterion.cuda()

#implement training routine
def train_model(num_epoch):
    conv_model.train()
    exp_lr_scheduler.step()

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.unsqueeze(1)
        data, target = data, target

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        output = conv_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                num_epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx + 1) / len(train_loader), loss.item()))


def evaluate(data_loader):
    global performance
    conv_model.eval()
    loss = 0
    correct = 0

    for data, target in data_loader:
        data = data.unsqueeze(1)
        data, target = data, target

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        output = conv_model(data)

        loss += F.cross_entropy(output, target, size_average=False).item()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    loss /= len(data_loader.dataset)

    performance.append(100. * correct / len(data_loader.dataset))

    print('\nAverage Val Loss: {:.4f}, Val Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))


#do training
num_epochs = 15
global performance
performance = []

for n in range(num_epochs):
    train_model(n)
    evaluate(val_loader)

#save the model
torch.save(conv_model.state_dict(), "D:/Programming/Python/SudokuSolver/data/moddedMNIST/model_complex_v2_ext.sav")

#create test routine
def make_predictions(data_loader):
    conv_model.eval()
    test_preds = torch.LongTensor()

    for i, data in enumerate(data_loader):
        data = data.unsqueeze(1)

        if torch.cuda.is_available():
            data = data.cuda()

        output = conv_model(data)

        preds = output.cpu().data.max(1, keepdim=True)[1]
        test_preds = torch.cat((test_preds, preds), dim=0)

    return test_preds

test_set_preds = make_predictions(test_loader)

conf_matrix = confusion_matrix(test_labels, test_set_preds)
print(conf_matrix)
result = classification_report(test_labels, test_set_preds)
print(result)

plt.plot(range(1, len(performance) +1), performance)
plt.ylabel('Accuracy')
plt.xlabel('Epoche')
plt.show()

print(accuracy_score(test_labels, test_set_preds))