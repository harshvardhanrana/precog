from torchvision.models import detection
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
f = open('../COCO.txt', 'r')
CLASSES = f.read().split('\n')
f.close()
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

model = detection.fasterrcnn_resnet50_fpn(weights = "FasterRCNN_ResNet50_FPN_Weights.DEFAULT").to(DEVICE)
model.eval()

threshold = 0.5

def get_objects(cv_image) -> list[tuple[int,int]]:
    image = cv_image.copy()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))

    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = torch.FloatTensor(image)

    image = image.to(DEVICE)
    detections = model(image)[0]

    objects_list = []

    for i in range(0, len(detections["boxes"])):

        confidence = detections["scores"][i]

        if confidence > threshold:
            idx = int(detections["labels"][i])
            objects_list.append((CLASSES[idx], confidence.item() * 100))
    
    return objects_list

def draw_objects(cv_image):
    image = cv_image.copy()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))

    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = torch.FloatTensor(image)

    image = image.to(DEVICE)
    detections = model(image)[0]

    objects_list = []

    for i in range(0, len(detections["boxes"])):

        confidence = detections["scores"][i]

        if confidence > threshold:
            idx = int(detections["labels"][i])
            objects_list.append((CLASSES[idx], confidence.item() * 100))
            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")
            label = f"{CLASSES[idx]}: {confidence * 100:.2f}"
            cv2.rectangle(cv_image, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(cv_image, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 2, COLORS[idx], 2)
    
    return cv_image