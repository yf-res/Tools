import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import precision_recall_curve, accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt

# Custom Dataset class
class TimeSequenceImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.FloatTensor(images)  # Shape: [N, 3, 32, 32]
        self.labels = torch.FloatTensor(labels)  # Shape: [N]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Simple CNN Architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Complex ResNet-style Architecture
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += self.shortcut(residual)
        x = torch.relu(x)
        return x

class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.res_blocks = nn.Sequential(
            ResBlock(32, 32),
            ResBlock(32, 64),
            nn.MaxPool2d(2),
            ResBlock(64, 64),
            ResBlock(64, 128),
            nn.MaxPool2d(2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.initial(x)
        x = self.res_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ModelTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        
    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images).squeeze()
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
        return running_loss / len(train_loader)
    
    def evaluate(self, val_loader, threshold=0.5):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                outputs = self.model(images).squeeze().cpu().numpy()
                all_preds.extend(outputs)
                all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics for different thresholds
        precisions, recalls, thresholds = precision_recall_curve(all_labels, all_preds)
        
        # Find best threshold based on F1 score
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-7)
        best_threshold = thresholds[np.argmax(f1_scores[:-1])]
        
        # Calculate metrics for both default and best thresholds
        metrics = {}
        for thresh in [threshold, best_threshold]:
            binary_preds = (all_preds >= thresh).astype(int)
            metrics[f'threshold_{thresh:.2f}'] = {
                'accuracy': accuracy_score(all_labels, binary_preds),
                'precision': precision_score(all_labels, binary_preds),
                'recall': recall_score(all_labels, binary_preds)
            }
        
        return metrics, all_preds, all_labels

def train_model(model, train_loader, val_loader, epochs=10):
    trainer = ModelTrainer(model)
    history = {'train_loss': [], 'val_metrics': []}
    
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_metrics, _, _ = trainer.evaluate(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_metrics'].append(val_metrics)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}')
        for thresh, metrics in val_metrics.items():
            print(f'Validation Metrics ({thresh}):')
            print(f'Accuracy: {metrics["accuracy"]:.4f}')
            print(f'Precision: {metrics["precision"]:.4f}')
            print(f'Recall: {metrics["recall"]:.4f}')
        print('-' * 50)
    
    return trainer, history

# Example usage
def main():
    # Create dummy data for demonstration
    batch_size = 32
    X = np.random.randn(100, 3, 32, 32)  # 100 images of size 32x32x3
    y = np.random.randint(0, 2, 100)     # Binary labels
    
    # Create datasets
    dataset = TimeSequenceImageDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Train Simple CNN
    print("Training Simple CNN...")
    simple_model = SimpleCNN()
    simple_trainer, simple_history = train_model(simple_model, train_loader, val_loader)
    
    # Train Complex CNN
    print("\nTraining Complex CNN...")
    complex_model = ComplexCNN()
    complex_trainer, complex_history = train_model(complex_model, train_loader, val_loader)

if __name__ == "__main__":
    main()
