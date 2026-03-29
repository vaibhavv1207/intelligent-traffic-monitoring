import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class AccidentDataset(Dataset):
    def __init__(self, root_dir, sequence_length=10, transform=None):
        self.sequence_length = sequence_length
        self.transform = transform
        self.samples = []  # list of (image_path, label)

        # Class mapping
        self.class_map = {}
        for label, folder in enumerate(sorted(os.listdir(root_dir))):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            self.class_map[folder] = label
            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((
                        os.path.join(folder_path, img_file),
                        label
                    ))

        print(f"Class mapping: {self.class_map}")
        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.FloatTensor(img).permute(2, 0, 1) / 255.0

        # Create a fake sequence by repeating the frame
        # In real deployment, actual video frames are used
        sequence = img.unsqueeze(0).repeat(self.sequence_length, 1, 1, 1)

        return sequence, label