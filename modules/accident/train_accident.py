import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from accident_detector import AccidentCNNLSTM
from accident_dataset import AccidentDataset

# Paths
TRAIN_DIR = r"C:\Users\vaibh\Desktop\intelligent-traffic-monitoring\data\accident_dataset\train"
VAL_DIR   = r"C:\Users\vaibh\Desktop\intelligent-traffic-monitoring\data\accident_dataset\val"
SAVE_PATH = r"C:\Users\vaibh\Desktop\intelligent-traffic-monitoring\modules\accident\accident_model.pth"

# Hyperparameters
EPOCHS       = 20
BATCH_SIZE   = 8
LR           = 0.0001
SEQ_LENGTH   = 10
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training on: {DEVICE}")

# Transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Datasets
train_dataset = AccidentDataset(TRAIN_DIR, SEQ_LENGTH, transform)
val_dataset   = AccidentDataset(VAL_DIR,   SEQ_LENGTH, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

# Model
model = AccidentCNNLSTM().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

best_val_acc = 0.0

for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for sequences, labels in train_loader:
        sequences = sequences.to(DEVICE)
        labels    = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    train_acc = 100.0 * correct / total

    # Validation
    model.eval()
    val_correct = 0
    val_total   = 0

    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(DEVICE)
            labels    = labels.to(DEVICE)
            outputs   = model(sequences)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total   += labels.size(0)

    val_acc = 100.0 * val_correct / val_total
    scheduler.step()

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {train_loss/len(train_loader):.4f} "
          f"Train Acc: {train_acc:.1f}% "
          f"Val Acc: {val_acc:.1f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  --> Best model saved! Val Acc: {val_acc:.1f}%")

print(f"\nTraining complete! Best Val Accuracy: {best_val_acc:.1f}%")
print(f"Model saved at: {SAVE_PATH}")