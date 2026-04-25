import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GATImageClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super(GATImageClassifier, self).__init__()
        # First GAT layer with multi-head attention
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        
        # Second GAT layer (output_channels * heads must match input for next layer or be averaged)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=0.6)
        
        # Linear layer for final classification after pooling
        self.classifier = torch.nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))

        # 2. Readout layer: Pool node features into a graph-level representation
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels * heads]

        # 3. Apply a final classifier
        return self.classifier(x)

    def train(self, data, optimizer, criterion):
        self.train()
        optimizer.zero_grad()
        out = self.forward(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        return loss.item()

model = GATImageClassifier(in_channels=3, hidden_channels=64, out_channels=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train(train_loader):
    model.train()
    for data in train_loader:
        optimizer.zero_grad()  # Clear gradients
        out = model(data.x, data.edge_index, data.batch) # Forward pass
        loss = criterion(out, data.y) # Compute loss
        loss.backward() # Backpropagation
        optimizer.step() # Update weights
