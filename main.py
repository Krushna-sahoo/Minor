import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.onnx
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import modal

# =======================
# 📌 Define Constants
# =======================
DATASET_PATH = r"C:\Users\KIIT\Documents\Minor\splitted_data"
BATCH_SIZE = 8  # Increased batch size
EPOCHS = 10
PATIENCE = 3  # Early stopping patience

# =======================
# 📌 Setup Modal GPU
# =======================
GPU_CONTAINER = modal.Image.debian_slim().apt_install(
    "libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev"
).pip_install(
    ["torch", "torchvision", "timm", "numpy", "scikit-learn", "opencv-python"]
)



stub = modal.App("deepfake-detection")
@stub.function(image=GPU_CONTAINER, gpu="T4")
def train_model():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # =======================
    # 📌 Data Augmentation & Dataset
    # =======================
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    class DeepFakeDataset(Dataset):
        def __init__(self, folder, label, transform=None, frames_per_video=30):
            self.folder = folder
            self.label = label
            self.transform = transform
            self.frames_per_video = frames_per_video
            self.video_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".mp4")]

        def __len__(self):
            return len(self.video_files)

        def __getitem__(self, idx):
            video_path = self.video_files[idx]
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            while len(frames) < self.frames_per_video and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frame = self.transform(frame)
                frames.append(frame)
            cap.release()
            
            while len(frames) < self.frames_per_video:
                frames.append(torch.zeros(3, 224, 224))
            
            return torch.stack(frames), torch.tensor(self.label, dtype=torch.long)

    # =======================
    # 📌 Load Dataset
    # =======================
    train_real = DeepFakeDataset(os.path.join(DATASET_PATH, "real/train"), label=0, transform=transform)
    train_fake = DeepFakeDataset(os.path.join(DATASET_PATH, "fake/train"), label=1, transform=transform)
    val_real = DeepFakeDataset(os.path.join(DATASET_PATH, "real/val"), label=0, transform=transform)
    val_fake = DeepFakeDataset(os.path.join(DATASET_PATH, "fake/val"), label=1, transform=transform)

    train_loader = DataLoader(train_real + train_fake, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_real + val_fake, batch_size=BATCH_SIZE, shuffle=False)

    # =======================
    # 📌 Define Model
    # =======================
    class DeepFakeDetector(nn.Module):
        def __init__(self):
            super(DeepFakeDetector, self).__init__()
            self.feature_extractor = timm.create_model("efficientnet_lite0", pretrained=True)
            self.feature_extractor.classifier = nn.Identity()
            self.lstm = nn.LSTM(input_size=1280, hidden_size=256, num_layers=1, batch_first=True, dropout=0.3)
            self.fc = nn.Linear(256, 2)

        def forward(self, x):
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            features = self.feature_extractor(x)
            features = features.view(B, T, -1)
            lstm_out, _ = self.lstm(features)
            output = self.fc(lstm_out[:, -1, :])
            return output

    # =======================
    # 📌 Train Model with Early Stopping & Learning Rate Scheduler
    # =======================
    model = DeepFakeDetector().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    stopping_rounds = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_preds, train_labels = 0, [], []
        for videos, labels in train_loader:
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(videos)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        train_acc = accuracy_score(train_labels, train_preds)
        
        model.eval()
        val_loss, val_preds, val_labels = 0, [], []
        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(DEVICE), labels.to(DEVICE)
                with torch.cuda.amp.autocast():
                    outputs = model(videos)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        val_acc = accuracy_score(val_labels, val_preds)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_deepfake_detector.pth")
            stopping_rounds = 0
        else:
            stopping_rounds += 1
            if stopping_rounds >= PATIENCE:
                print("Early stopping triggered! Stopping training.")
                break

if __name__ == "__main__":
    with stub.run():
        train_model.remote()
