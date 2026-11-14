import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class PyTorchLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(PyTorchLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # BatchFirst=True -> (batch, seq, feature)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # h0, c0 defaults to zeros if not provided
        out, (hn, cn) = self.lstm(x)
        
        # We typically want the output of the last time step
        # out shape: (batch_size, seq_len, hidden_dim)
        last_out = out[:, -1, :] 
        
        out = self.fc(last_out)
        return out

class LstmWrapper:
    def __init__(self, window_size=10, hidden_dim=50, num_layers=1, dropout=0.2, epochs=50, batch_size=32, learning_rate=0.001):
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_dim = 0 # Calculated: TotalCols / window_size
        
    def _reshape_input(self, X):
        # X is (N, TotalCols)
        # We need (N, WindowSize, FeatureDim)
        # TotalCols must be divisible by WindowSize
        total_cols = X.shape[1]
        if total_cols % self.window_size != 0:
            raise ValueError(f"Feature count ({total_cols}) is not divisible by window_size ({self.window_size}). Cannot reshape for LSTM.")
        
        self.feature_dim = total_cols // self.window_size
        return X.reshape(X.shape[0], self.window_size, self.feature_dim)

    def fit(self, X, y):
        X_reshaped = self._reshape_input(X)
        self.classes_ = np.unique(y)
        output_dim = len(self.classes_)
        
        X_tensor = torch.tensor(X_reshaped, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        
        self.model = PyTorchLSTM(self.feature_dim, self.hidden_dim, self.num_layers, output_dim, self.dropout).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        print(f"[PyTorch LSTM] Training on {self.device} with Window={self.window_size}, FeatureDim={self.feature_dim}, Hidden={self.hidden_dim}...")
        
        self.model.train()
        for epoch in range(self.epochs):
            if callable(getattr(self, 'cancel_check', None)) and self.cancel_check():
                raise InterruptedError("LSTM training cancelled")
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                if callable(getattr(self, 'cancel_check', None)) and self.cancel_check():
                    raise InterruptedError("LSTM training cancelled")
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.epochs} Loss: {epoch_loss / len(dataloader):.4f}")

    def predict(self, X):
        self.model.eval()
        X_reshaped = self._reshape_input(X)
        X_tensor = torch.tensor(X_reshaped, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        self.model.eval()
        X_reshaped = self._reshape_input(X)
        X_tensor = torch.tensor(X_reshaped, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()

# Adapter for gpu_server.py
def train(X, y, params):
    # Parse params
    window_size = int(params.get('window_size', 1))
    hidden_dim = int(params.get('hidden_dim', 64))
    num_layers = int(params.get('num_layers', 2))
    dropout = float(params.get('dropout', 0.2))
    epochs = int(params.get('epochs', 50)) # map to max_iter
    if 'max_iter' in params: epochs = int(params['max_iter'])
    batch_size = int(params.get('batch_size', 32))
    learning_rate = float(params.get('learning_rate', 0.001))
    
    wrapper = LstmWrapper(window_size, hidden_dim, num_layers, dropout, epochs, batch_size, learning_rate)
    wrapper.cancel_check = params.get('cancel_check')
    wrapper.fit(X, y)
    return wrapper
