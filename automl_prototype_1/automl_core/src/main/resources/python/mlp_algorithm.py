import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

class PyTorchMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation='relu'):
        super(PyTorchMLP, self).__init__()
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'logistic':
                layers.append(nn.Sigmoid())
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class MlpWrapper:
    def __init__(self, hidden_layers=(100,), activation='relu', solver='adam', alpha=0.0001, max_iter=200, batch_size=32, learning_rate=0.001):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dim = 0
        self.classes_ = None

    def fit(self, X, y):
        # Convert to Tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        
        self.classes_ = np.unique(y)
        self.output_dim = len(self.classes_)
        
        # Remap labels to 0..N-1 if needed
        # (For now assume labels are already 0-indexed integers or handle mapping)
        
        input_dim = X.shape[1]
        
        self.model = PyTorchMLP(input_dim, self.hidden_layers, self.output_dim, self.activation).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = None
        if self.solver == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.alpha)
        elif self.solver == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.alpha)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate) # Default
            
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        print(f"[PyTorch MLP] Training on {self.device} with {self.hidden_layers} hidden units...")
        
        self.model.train()
        for epoch in range(self.max_iter):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.max_iter} Loss: {epoch_loss / len(dataloader):.4f}")

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()
        
    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()

# Adapter for gpu_server.py
def train(X, y, params):
    # Parse params
    layers_str = str(params.get('hidden_layer_sizes', "100")).strip()
    try:
        hidden_layers = tuple(map(int, layers_str.split(',')))
    except:
        hidden_layers = (100,)
        
    activation = params.get('activation', 'relu')
    solver = params.get('solver', 'adam')
    alpha = float(params.get('alpha', 0.0001))
    max_iter = int(params.get('max_iter', 200))
    
    wrapper = MlpWrapper(hidden_layers=hidden_layers, activation=activation, solver=solver, alpha=alpha, max_iter=max_iter)
    wrapper.fit(X, y)
    return wrapper
