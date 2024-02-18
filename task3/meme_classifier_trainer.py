import torch
import torchvision

import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn



import time
from pytorch_dataset import CustomImageDataset
from cnn import Net

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [transforms.Resize((150, 150)), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

training_data = CustomImageDataset('../new_data_set/dataset_info.txt', '../new_data_set/', transform=transform)
trainloader = torch.utils.data.DataLoader(training_data, batch_size=50, shuffle=True)

net = Net().to(DEVICE)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

start = time.time()

for epoch in range(15):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs).to(DEVICE)
        loss = criterion(outputs, labels).to(DEVICE)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # if i % 50 == 49:    # print every 2000 mini-batches
        #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        #     running_loss = 0.0
    print(running_loss)

print('Finished Training')

PATH = './meme_net.pth'
torch.save(net.state_dict(), PATH)

end = time.time()
print(end - start)