import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLandmarkOnlyCNN(nn.Module):
    def __init__(self, landmark_dim, out_channels, depth_map_size):
        super(SimpleLandmarkOnlyCNN, self).__init__()
        self.flatten_size = landmark_dim
        self.fc1 = nn.Linear(landmark_dim, self.flatten_size)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=depth_map_size)
        self.conv_final = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, landmarks):
        print(landmarks)
        batch_size = landmarks.size(0)
        print(batch_size)
        x = F.relu(self.fc1(landmarks))
        x = x.view(batch_size, 1, 85, 85)  # Ensure this matches the depth_map_size
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.conv_final(x)
        return x
