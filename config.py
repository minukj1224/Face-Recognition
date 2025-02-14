import torch
import os
from model import ResNet18, ResNet34, ResNet50, ResNet101, ArcFaceLoss

DATASET_PATH = r"C:\Users\Minuk\Desktop\face_recognition\dataset\img"
TRAIN_RATIO = 0.9
TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, "train")
VAL_DATASET_PATH = os.path.join(DATASET_PATH, "val")

IMG_SIZE = (96, 96)
AUG_MULTIPLIER = 50
BATCH_SIZE = 32

MODEL = ResNet101
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SCHEDULER_STEP_SIZE = 10
SCHEDULER_GAMMA = 0.5
CHECKPOINT_INTERVAL = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = None

MODEL_SAVE_PATH = r"C:\Users\Minuk\Desktop\face_recognition\models"
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, "best_model.pth")
LAST_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, "last_model.pth")

LOG_PATH = os.path.join(MODEL_SAVE_PATH, "logs")
os.makedirs(LOG_PATH, exist_ok=True)

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

AUGMENTATION_ENABLED = True