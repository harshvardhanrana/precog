from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
import torch
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
f = open('imagenetclasses.txt', 'r')
CLASSES = f.read().split('\n')
f.close()

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights = weights).to(DEVICE)
model.eval()

preprocess = weights.transforms()

def get_objects(cv_image):

    # image = read_image(img_path).to(DEVICE)
    image = transforms.ToTensor()(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)).to(DEVICE)
    batch = preprocess(image).unsqueeze(0)

    prediction = model(batch)

    _, indices = torch.sort(prediction, descending=True)
    percentage = torch.nn.functional.softmax(prediction, dim=1)[0] * 100
    return [(CLASSES[idx], percentage[idx].item()) for idx in indices[0][:5] if percentage[idx].item() > 2]
