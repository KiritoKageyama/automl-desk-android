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
    def __init__(self, epochs=10, batch_size=32, lr=0.001, augment=None):
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.augment = augment or {}
        self.model = None
        self.cancel_check = None
        self.train_image_paths = None
        self._infer_paths = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # SimpleCNN expects 64x64 tensors after two pool layers -> 16x16
        ops = []
        if self.augment.get('flip'):
            ops.append(transforms.RandomHorizontalFlip())
        if self.augment.get('rotate'):
            ops.append(transforms.RandomRotation(15))
        if self.augment.get('crop'):
            ops.append(transforms.RandomResizedCrop(64, scale=(0.8, 1.0)))
        ops.append(transforms.Resize((64, 64)))
        ops.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform = transforms.Compose(ops)
        # Eval transform without random augments
        self.eval_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def _resolve_paths(self, X):
        if getattr(self, '_infer_paths', None):
            paths = list(self._infer_paths)
            self._infer_paths = None
            return paths
        if isinstance(X, (list, tuple)) and len(X) > 0 and isinstance(X[0], str):
            return list(X)
        arr = np.asarray(X)
        if arr.dtype == object and arr.size > 0:
            return [str(x) for x in arr.ravel()]
        # Numeric placeholder features from Java ImageDataProvider (n x 0 or n x k zeros)
        if self.train_image_paths is not None and len(arr) == len(self.train_image_paths):
            print("[CNN] Using stored train image paths for predict (numeric feature placeholders).")
            return list(self.train_image_paths)
        raise RuntimeError(
            "CNN predict requires image file paths; received numeric features without paths. "
            "Pass image_paths in the predict job or train with image_paths params."
        )

    def fit(self, X, y, params=None):
        params = params or {}
        image_paths = params.get('image_paths')

        if image_paths:
            print(f"[CNN] Training with {len(image_paths)} image paths. backbone={params.get('backbone')} strategy={params.get('strategy')}")
            if params.get('backbone') and str(params.get('backbone')).upper() not in ('', 'CUSTOM', 'CUSTOM CNN (3 BLOCK)', 'SIMPLECNN'):
                print(f"[CNN] NOTE: backbone '{params.get('backbone')}' is logged only; training uses SimpleCNN @ 64x64.")
            paths = list(image_paths)
            dataset = ImagePathDataset(paths, y, transform=self.transform)
        elif isinstance(X, list) and len(X) > 0 and isinstance(X[0], str):
            print(f"[CNN] Training with {len(X)} image paths (passed as X).")
            paths = list(X)
            dataset = ImagePathDataset(paths, y, transform=self.transform)
        else:
            print("[CNN] Error: No image paths provided. Passing raw features not yet supported for CNN.")
            return self

        self.train_image_paths = paths
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        num_classes = len(np.unique(y))

        self.model = SimpleCNN(num_classes=num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            if callable(self.cancel_check) and self.cancel_check():
                raise InterruptedError("Training cancelled by user")
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
        if self.model is None:
            raise RuntimeError("CNN model not trained")
        paths = self._resolve_paths(X)
        dataset = ImagePathDataset(paths, [0] * len(paths), transform=self.eval_transform)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        preds = []
        self.model.eval()
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
                preds.extend(batch_preds.tolist())
        return np.asarray(preds, dtype=int)

# Entry points
def train(X, y, params):
    # Requested image_size is honored as preprocess intent; model currently trains at 64x64
    if params.get('image_size'):
        print(f"[CNN] Requested image_size={params.get('image_size')} (internal train size fixed at 64x64 for SimpleCNN)")

    clf = CnnWrapper(
        epochs=params.get('epochs', 10),
        batch_size=params.get('batch_size', 32),
        lr=params.get('learning_rate', 0.001),
        augment={
            'flip': bool(params.get('augment_flip', False)),
            'rotate': bool(params.get('augment_rotate', False)),
            'crop': bool(params.get('augment_crop', False)),
        }
    )
    clf.cancel_check = params.get('cancel_check')
    clf.fit(X, y, params)
    return clf

def fit(X, y, params):
    return train(X, y, params)
