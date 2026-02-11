
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin

class SimpleFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleFeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return self.softmax(out)

class PyTorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_dim=64, epochs=50, lr=0.01):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, X, y, params=None):
        if params:
            self.hidden_dim = int(params.get('hidden_dim', self.hidden_dim))
            self.epochs = int(params.get('epochs', self.epochs))
            self.lr = float(params.get('lr', self.lr))
            
        print(f"[PyTorch Script] Training on {self.device} with hidden_dim={self.hidden_dim}, epochs={self.epochs}")
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        
        input_dim = X.shape[1]
        output_dim = len(np.unique(y))
        
        self.model = SimpleFeedForward(input_dim, self.hidden_dim, output_dim).to(self.device)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 10 == 0:
                 print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")
                 
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.cpu().numpy()

# Standard Interface for AutoML Server
def train(X, y, params):
    clf = PyTorchClassifier()
    clf.fit(X, y, params)
    return clf

def fit(X, y, params):
    return train(X, y, params)
