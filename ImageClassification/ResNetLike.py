class ResNetInspiredCNN(nn.Module):
    def __init__(self):
        super(ResNetInspiredCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x2 = self.pool(x2)
        
        # Skip connection: Adding input of conv2 to output of conv3
        x3 = F.relu(self.conv3(x2)) + x2
        x3 = self.pool(x3)
        
        x3 = x3.view(-1, 32 * 8 * 8)  # Flatten
        x3 = F.relu(self.fc1(x3))
        x3 = torch.sigmoid(self.fc2(x3))
        return x3

# Example usage
model = ResNetInspiredCNN()
criterion = nn.BCELoss()
