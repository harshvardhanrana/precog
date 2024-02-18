import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision.io import read_image

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 100)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 10 * 10, 84)
        self.fc2 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

PATH = '../task3/meme_net.pth'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = Net().to(DEVICE)
net.load_state_dict(torch.load(PATH))

net.eval()

transform = transforms.Compose(
    [transforms.Resize((150, 150)), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def is_meme_or_not(orig_img) -> tuple[int, int]:
    cv_img = orig_img.copy()
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    tensor_img = transforms.ToTensor()(cv_img)
    tensor_img = transform(tensor_img).unsqueeze(0).to(DEVICE)

    output = net(tensor_img).to(DEVICE)

    _, predicted = torch.max(output.data, 1)
    return (predicted.item(), _.item())