import torch
import torch.nn as nn
import torch.nn.functional as F

class LandmarkToDepthCNN(nn.Module):
    def __init__(self, landmark_dim=468, landmark_coords=3, out_channels=1, depth_map_size=(85, 85)):
        super(LandmarkToDepthCNN, self).__init__()

        self.flatten_size = landmark_dim * landmark_coords

        # Feature extraction
        self.fc1 = nn.Linear(self.flatten_size, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 16 * depth_map_size[0] * depth_map_size[1])

        # Reshape layer
        self.reshape = nn.Unflatten(1, (16, depth_map_size[0], depth_map_size[1]))

        # Convolutional layers
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.AdaptiveAvgPool2d(depth_map_size)

    def forward(self, landmarks):
        batch_size = landmarks.size(0)

        # Flatten and pass through fully connected layers
        x = landmarks.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Reshape for convolutional layers
        x = self.reshape(x)

        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        # Pooling (optional, depending on your depth map size)
        x = self.pool(x)

        return x
