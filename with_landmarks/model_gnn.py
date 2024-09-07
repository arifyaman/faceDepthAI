import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class SimpleGNN(nn.Module):
    def __init__(self, landmark_dim=478, out_channels=1, depth_map_size=(640, 360)):
        super(SimpleGNN, self).__init__()
        
        self.depth_map_size = depth_map_size
        
        # Graph Convolutional Layers
        self.conv1 = GCNConv(landmark_dim * 3, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 32)
        
        # Fully Connected Layers for output
        self.fc1 = nn.Linear(32, 256)
        self.fc2 = nn.Linear(256, depth_map_size[0] * depth_map_size[1])
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, edge_index):
        # Apply graph convolution layers
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        
        # Pooling and fully connected layers
        x = torch.mean(x, dim=0)  # Global mean pooling
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        # Reshape to match depth map size
        x = x.view(-1, 1, self.depth_map_size[0], self.depth_map_size[1])
        
        return x

# Example usage
# Assuming you have edge_index and x (node features) prepared
# model = SimpleGNN(landmark_dim=478, out_channels=1, depth_map_size=(640, 360))
