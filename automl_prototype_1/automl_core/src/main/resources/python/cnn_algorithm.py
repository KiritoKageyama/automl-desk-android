
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# --- Custom Dataset for efficient loading ---
class ImagePathDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Create black image on failure
            image = Image.new('RGB', (64, 64), color='black')
            
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        return image, label

# --- CNN Model Class ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128) # Assumes 64x64 input
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16) 
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CnnWrapper:
    def __init__(self, epochs=10, batch_size=32, lr=0.001):
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])

    def fit(self, X, y, params=None):
        # Check if X contains paths or raw data
        # If parameters has 'image_paths', use that
        image_paths = params.get('image_paths') if params else None
        
        if image_paths:
            # Case 1: Training from image paths
            print(f"[CNN] Training with {len(image_paths)} image paths.")
            dataset = ImagePathDataset(image_paths, y, transform=self.transform)
        else:
             # Case 2: X is potentially raw pixel data? 
             # For now, assume we rely on paths for Phase 3.
             if isinstance(X, list) and len(X) > 0 and isinstance(X[0], str):
                 # X itself is paths
                 print(f"[CNN] Training with {len(X)} image paths (passed as X).")
                 dataset = ImagePathDataset(X, y, transform=self.transform)
             else:
                 print("[CNN] Error: No image paths provided. Passing raw features not yet supported for CNN.")
                 return self

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        num_classes = len(np.unique(y))
        
        self.model = SimpleCNN(num_classes=num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {running_loss/len(dataloader):.4f}")
            
        return self

    def predict(self, X):
        self.model.eval()
        # X might be paths or data
        # Warning: For prediction we need paths again if we trained on paths
        # We assume X contains paths if it's a list of strings
        # But Java might pass dummy double[][] if we don't handle it.
        
        # Actually, in PythonExecutionService, we pass 'features' from Dataset.
        # If Dataset.features is empty, we must rely on 'image_paths' param or similar.
        # But predict() in IClassifier only takes features.
        # We need to solve this data passing issue.
        
        # Temporary workaround: Expect image paths in X (casted/converted on Java side? No)
        # Or Expect 'image_paths' in params? predict() doesn't take params.
        
        # Robust solution: CnnClassifier.predict() needs to fetch paths from Dataset if features are empty?
        # But predict() takes features as arg.
        
        # For this prototype: We will assume X is passed as a list of strings (using object array) 
        # OR we rely on the fact that for now we might be testing with loaded valid X.
        
        # Let's assume input X is standard. If not, we fail gracefully.
        return np.zeros(len(X), dtype=int) # Placeholder

# Entry points
def train(X, y, params):
    clf = CnnWrapper(
        epochs=params.get('epochs', 10),
        batch_size=params.get('batch_size', 32),
        lr=params.get('learning_rate', 0.001)
    )
    clf.fit(X, y, params)
    return clf

def fit(X, y, params):
    return train(X, y, params)
