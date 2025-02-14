import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import random
import numpy as np
import cv2
from model import ResNet18, ResNet34, ResNet50, ResNet101, ArcFaceLoss
from data import get_dataloaders
import config

train_loader, val_loader = get_dataloaders()

model = config.MODEL().to(config.DEVICE)
arcface_loss = ArcFaceLoss(128, len(train_loader.dataset.dataset.labels)).to(config.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.SCHEDULER_STEP_SIZE, gamma=config.SCHEDULER_GAMMA)

best_loss = float('inf')
os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(os.path.join(config.MODEL_SAVE_PATH, "test_results"), exist_ok=True)

def extract_feature(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, config.IMG_SIZE)
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
    img = img.unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        feature = model(img).cpu().numpy().flatten()
    return feature / np.linalg.norm(feature)

def save_test_results(epoch):
    dataset_path = config.DATASET_PATH
    image_list = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png'))]
    if len(image_list) < 3:
        print("Not enough images for testing.")
        return

    test_img = random.choice(image_list)
    test_id = test_img.split('_')[0]
    same_img = random.choice([img for img in image_list if img.startswith(test_id) and img != test_img])
    diff_img = random.choice([img for img in image_list if not img.startswith(test_id)])

    test_path = os.path.join(dataset_path, test_img)
    same_path = os.path.join(dataset_path, same_img)
    diff_path = os.path.join(dataset_path, diff_img)

    test_feature = extract_feature(test_path)
    same_feature = extract_feature(same_path)
    diff_feature = extract_feature(diff_path)

    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    score_same = cosine_similarity(test_feature, same_feature)
    score_diff = cosine_similarity(test_feature, diff_feature)

    results_path = os.path.join(config.MODEL_SAVE_PATH, "test_results", f"epoch_{epoch+1}")
    os.makedirs(results_path, exist_ok=True)

    cv2.imwrite(os.path.join(results_path, "test.jpg"), cv2.imread(test_path))
    cv2.imwrite(os.path.join(results_path, "same.jpg"), cv2.imread(same_path))
    cv2.imwrite(os.path.join(results_path, "diff.jpg"), cv2.imread(diff_path))

    with open(os.path.join(results_path, "scores.txt"), "w") as f:
        f.write(f"Test Image: {test_img}\n")
        f.write(f"Same Image: {same_img}, Score: {score_same:.4f}\n")
        f.write(f"Different Image: {diff_img}, Score: {score_diff:.4f}\n")

    print(f"📄 Saved test results for epoch {epoch+1}.")

for epoch in range(config.EPOCHS):
    model.train()
    running_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config.EPOCHS}]")
    
    for images, labels in loop:
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
        optimizer.zero_grad()
        embeddings = model(images)
        loss = arcface_loss(embeddings, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Training Loss: {avg_loss:.4f}")

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            embeddings = model(images)
            loss = arcface_loss(embeddings, labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}")

    scheduler.step()
    
    torch.save(model.state_dict(), config.LAST_MODEL_PATH)
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), config.BEST_MODEL_PATH)
        print("✅ Best model updated!")
    
    if (epoch + 1) % config.CHECKPOINT_INTERVAL == 0:
        checkpoint_path = os.path.join(config.MODEL_SAVE_PATH, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"📌 Checkpoint saved: {checkpoint_path}")
    
    save_test_results(epoch)