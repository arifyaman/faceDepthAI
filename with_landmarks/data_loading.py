import numpy as np
import torch
from torch.utils.data import Dataset
import OpenEXR
import Imath
import os

def load_exr_depth_map(exr_path):
    try:
        # Open the EXR file
        exr_file = OpenEXR.InputFile(exr_path)
        
        # Get the header to extract image size
        header = exr_file.header()
        dw = header['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)  # (width, height)
        
        # Define the channel to extract (usually 'Y' for depth)
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        
        # Extract the Y channel
        depth_map_str = exr_file.channel('Y', FLOAT)
        
        # Convert the string data to numpy array
        depth_map = np.frombuffer(depth_map_str, dtype=np.float32).reshape((size[1], size[0]))
        
        return depth_map
    except Exception as e:
        print(f"Error loading EXR file {exr_path}: {e}")
        return np.zeros((85, 85))  # Return a default value or handle appropriately

class FacialDepthDatasetWithLandmarks(Dataset):
    def __init__(self, landmark_dir, depth_dir):
        self.landmark_dir = landmark_dir
        self.depth_dir = depth_dir
        self.file_names = [f.replace('.npy', '') for f in os.listdir(landmark_dir) if f.endswith('.npy')]
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        # Get the base filename
        base_name = self.file_names[idx]

        # Load the landmarks
        landmark_name = os.path.join(self.landmark_dir, base_name + '.npy')
        if not os.path.exists(landmark_name):
            raise FileNotFoundError(f"Landmark file {landmark_name} not found.")
        
        # Ensure landmarks are loaded as numeric array
        landmarks = np.load(landmark_name, allow_pickle=False).astype(np.float32)
        
        # Load the depth map
        depth_name = os.path.join(self.depth_dir, base_name + '.exr')
        if not os.path.exists(depth_name):
            raise FileNotFoundError(f"Depth map file {depth_name} not found.")
        depth_map = load_exr_depth_map(depth_name)
        
        # Convert landmarks and depth map to tensors
        landmarks = torch.tensor(landmarks, dtype=torch.float32)
        depth_map = torch.tensor(depth_map, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return landmarks, depth_map
