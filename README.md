# Face Recognition with ArcFace

This project implements a face recognition system using a ResNet-based deep learning model and the ArcFace loss function.

## Project Structure
- `config.py` - Configuration file for hyperparameters, paths, and settings.
- `data.py` - Dataset loading and augmentation logic.
- `model.py` - ResNet-based architecture with ArcFace loss.
- `train.py` - Training loop with model saving and epoch-wise evaluation.
- `test.py` - Face feature extraction and similarity comparison.
- `util.py` - Utility functions for image processing and feature extraction.

## Features
- **ArcFace Loss** for better face feature embedding.
- **ResNet18, 34, 50, 101** architectures implemented from scratch.
- **Data Augmentation** to increase dataset size and diversity.
- **Epoch-wise Evaluation** with saved test results (images and similarity scores).

## Training Instructions
1. Place your dataset in `DATASET_PATH` with image filenames containing the person's identity.
2. Configure parameters in `config.py`.
3. Run `train.py` to start training.
4. After each epoch, test results are saved in `test_results/`.

## Usage
- Extract face features using `test.py`.
- Compare two faces to calculate similarity scores.

## Requirements
- Python 3.8+
- PyTorch, Torchvision, OpenCV, Numpy

## Acknowledgment
This project is built upon PyTorch and inspired by ArcFace paper.

---