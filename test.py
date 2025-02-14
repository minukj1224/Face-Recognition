import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import cv2
import numpy as np
from tqdm import tqdm
import config

def extract_feature(img_path, model):
    model.eval()
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, config.IMG_SIZE)
    img = transform(img).unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        feature = model(img).cpu().numpy().flatten()
    return feature / np.linalg.norm(feature)

model.load_state_dict(torch.load(config.BEST_MODEL_PATH))
test_img = "C:\\Users\\Minuk\\Desktop\\face_test.jpg"
feature_vector = extract_feature(test_img, model)
print("Feature Vector:", feature_vector)