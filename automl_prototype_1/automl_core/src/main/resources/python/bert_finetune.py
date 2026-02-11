import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
import numpy as np
import time

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BertWrapper:
    def __init__(self, model_name='distilbert-base-uncased', epochs=3, batch_size=16, learning_rate=2e-5, params=None):
        self.params = params if params else {}
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.classes_ = None

    def fit(self, X, y):
        # X is list of strings (paths) or raw text?
        # If paths, load them. If raw strings in numpy array, use them.
        
        # Assumption: X is a numpy array of strings (raw text) OR file paths
        # If the first element looks like a file path, load content.
        
        texts = []
        if isinstance(X[0], str) and len(X[0]) < 256 and (X[0].endswith('.txt') or '/' in X[0] or '\\' in X[0]):
             # Likely paths
             import os
             print("[BERT] Loading texts from files...")
             for path in X:
                 if os.path.exists(path):
                     with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                         texts.append(f.read())
                 else:
                     texts.append("")
        else:
            # Assume raw text
            texts = X
            
        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)
        
        # Check for Custom Model ID
        self.model_name = self.params.get('hf_model_id', self.model_name)
        
        print(f"[BERT] Initializing {self.model_name} for {num_classes} classes on {self.device}...")
        
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Check for Custom Architecture / Scratch Training
        is_scratch = str(self.params.get('scratch', 'false')).lower() == 'true'
        
        if is_scratch:
            print(f"[BERT] Training from SCRATCH with custom config.")
            from transformers import DistilBertConfig, BertConfig, AutoConfig
            
            # Simple heuristic: if it looks like BERT, use DistilBertConfig for now or load config from pretrained but query architecture
            # For true generic scratch training, we need to know the Config class.
            # Allowing scratch on generic models is complex. We will stick to DistilBERT for scratch 
            # OR try to load config.
            
            try:
                config = AutoConfig.from_pretrained(self.model_name)
                # Override props
                if self.params.get('num_layers'): config.num_hidden_layers = int(self.params.get('num_layers'))
                if self.params.get('hidden_size'): config.hidden_size = int(self.params.get('hidden_size'))
                if self.params.get('num_heads'): config.num_attention_heads = int(self.params.get('num_heads'))
                config.num_labels = num_classes
                
                self.model = AutoModelForSequenceClassification.from_config(config).to(self.device)
                print(f"[BERT] Loaded Config from {self.model_name} and re-initialized weights.")
            except Exception as e:
                print(f"[BERT] Failed to load config for {self.model_name}, falling back to default DistilBERT config: {e}")
                # Fallback logic...
                config = DistilBertConfig(num_labels=num_classes)
                self.model = DistilBertForSequenceClassification(config).to(self.device)

        else:
            print(f"[BERT] Loading Pre-trained weights: {self.model_name}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=num_classes
            ).to(self.device)
        
        dataset = TextDataset(texts, y, self.tokenizer)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            start = time.time()
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.model.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                
            print(f"Epoch {epoch+1}/{self.epochs} Loss: {total_loss/len(loader):.4f} Time: {time.time()-start:.1f}s")

    def predict(self, X):
        self.model.eval()
        texts = []
        if isinstance(X[0], str) and len(X[0]) < 256 and (X[0].endswith('.txt') or '/' in X[0] or '\\' in X[0]):
             import os
             for path in X:
                 if os.path.exists(path):
                     with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                         texts.append(f.read())
                 else:
                     texts.append("")
        else:
            texts = X

        predictions = []
        # Batch prediction
        dataset = TextDataset(texts, [0]*len(texts), self.tokenizer) # dummy labels
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                
        return np.array(predictions)

def train(X, y, params):
    epochs = int(params.get('epochs', 3))
    batch_size = int(params.get('batch_size', 16))
    
    wrapper = BertWrapper(epochs=epochs, batch_size=batch_size, params=params)
    wrapper.fit(X, y)
    return wrapper
