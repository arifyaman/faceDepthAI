import torch
import torch.nn as nn
import torch.nn.functional as F

class LandmarkToDepthCNN(nn.Module):
    def __init__(self, landmark_dim=468, landmark_coords=3, out_channels=1, depth_map_size=(85, 85)):
        super(LandmarkToDepthCNN, self).__init__()

        # Flatten input landmarks (468 x 3)
        self.flatten_size = landmark_dim * landmark_coords
        
        # Fully connected layers to process landmark input
        self.fc1 = nn.Linear(self.flatten_size, 4096)  # Adjustable layer size
        self.fc2 = nn.Linear(4096, depth_map_size[0] * depth_map_size[1] * 16)  # Expand to fit depth map

        # Convolutional layers to refine the depth map
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, padding=1)

        # Pooling layer (optional)
        self.pool = nn.AdaptiveAvgPool2d(output_size=depth_map_size)

    def forward(self, landmarks):
        batch_size = landmarks.size(0)

        # Flatten the landmarks (batch_size, 468, 3) -> (batch_size, 468*3)
        x = landmarks.view(batch_size, -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Reshape to (batch_size, 16, 85, 85) for the convolutional layers
        x = x.view(batch_size, 16, 85, 85)

        # Convolutional layers with relu activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Final convolution to produce the depth map
        x = self.conv3(x)

        return x
