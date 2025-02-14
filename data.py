# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# import os
# import cv2
# import numpy as np
# from tqdm import tqdm
# import config
# from torch.utils.data import Dataset, DataLoader

# class FaceDataset(Dataset):
#     def __init__(self, dataset_path, transform=None):
#         self.dataset_path = dataset_path
#         self.transform = transform
#         self.data = []
        
#         for file in os.listdir(dataset_path):
#             if file.endswith(".jpg") or file.endswith(".png"):
#                 identity = file.split("_")[0]
#                 self.data.append((os.path.join(dataset_path, file), identity))
        
#         self.labels = {identity: idx for idx, identity in enumerate(sorted(set([d[1] for d in self.data])))}
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         img_path, identity = self.data[idx]
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, config.IMG_SIZE)
        
#         if self.transform:
#             img = self.transform(img)
        
#         label = self.labels[identity]
#         return img, torch.tensor(label, dtype=torch.long)

# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize(config.IMG_SIZE),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])
# ])

# def get_dataloaders():
#     dataset = FaceDataset(config.DATASET_PATH, transform=transform)
#     train_size = int(config.TRAIN_RATIO * len(dataset))
#     val_size = len(dataset) - train_size
#     train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
#     train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
#     return train_loader, val_loader

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import cv2
import numpy as np
from tqdm import tqdm
import config
from torch.utils.data import Dataset, DataLoader
import random

class FaceDataset(Dataset):
    def __init__(self, dataset_path, transform=None, augmentation=False):
        self.dataset_path = dataset_path
        self.transform = transform
        self.augmentation = augmentation
        self.data = []
        
        for file in os.listdir(dataset_path):
            if file.endswith(".jpg") or file.endswith(".png"):
                identity = file.split("_")[0]
                self.data.append((os.path.join(dataset_path, file), identity))
        
        self.labels = {identity: idx for idx, identity in enumerate(sorted(set([d[1] for d in self.data])))}

        self.aug_multiplier = config.AUG_MULTIPLIER if self.augmentation else 1
    
    def __len__(self):
        return len(self.data) * self.aug_multiplier
    
    def __getitem__(self, idx):
        original_idx = idx % len(self.data)
        img_path, identity = self.data[original_idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, config.IMG_SIZE)
        
        if self.transform:
            img = self.transform(img)

        if self.augmentation:
            img = self.apply_augmentation(img)
        
        label = self.labels[identity]
        return img, torch.tensor(label, dtype=torch.long)
    
    def apply_augmentation(self, img):
        aug_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        ])
        return aug_transforms(img)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def get_dataloaders():
    dataset = FaceDataset(config.DATASET_PATH, transform=transform, augmentation=True)
    train_size = int(config.TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader