import torch
import torchvision.transforms as transforms
import torchvision
from pytorch_dataset import CustomImageDataset
from cnn import Net
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

PATH = './meme_net.pth'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [transforms.Resize((150, 150)), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testing_data = CustomImageDataset('../new_data_set/dataset_info.txt', '../new_data_set/', transform=transform)
testloader = torch.utils.data.DataLoader(testing_data, batch_size=50, shuffle=True)

net = Net().to(DEVICE)

net.load_state_dict(torch.load(PATH))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join(f'{labels[j]:5s}' for j in range(4)))
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        # calculate outputs by running images through the network
        outputs = net(images).to(DEVICE)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 50 test images: {100 * correct // total} %')