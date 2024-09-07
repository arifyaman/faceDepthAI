import OpenEXR
import Imath
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt

def load_exr_depth_map(exr_path):
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

class FacialDepthDataset(Dataset):
    def __init__(self, image_dir, landmark_dir, depth_dir, transform=None):
        self.image_dir = image_dir
        self.landmark_dir = landmark_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]  # Ensure only .jpg files
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Load the image
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        image = cv2.imread(img_name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV default) to RGB
        
        # Load the landmarks
        #landmark_name = os.path.join(self.landmark_dir, self.image_filenames[idx].replace('.jpg', '.npy'))
        #landmarks = np.load(landmark_name)
        
        # Load the depth map
        depth_name = os.path.join(self.depth_dir, self.image_filenames[idx].replace('.jpg', '.exr'))
        depth_map = load_exr_depth_map(depth_name)
        
        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)
        
        # Convert landmarks and depth map to tensors
        #landmarks = torch.tensor(landmarks, dtype=torch.float32)
        depth_map = torch.tensor(depth_map, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return image, depth_map
 
# Define transformations for the images
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize between -1 and 1
])
print("Data loaded!")
# # Create the dataset
# dataset = FacialDepthDataset('without_background_images/', 'landmarks/', 'normalized_depth_maps/', transform=transform)

# # Split dataset into training and validation sets
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# # Create data loaders
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# # Test the data loader
#     # Display the first image, landmarks, and depth map
#     plt.figure(figsize=(15, 5))
    
#     # Display image
#     plt.subplot(1, 3, 1)
#     plt.imshow(images[0].permute(1, 2, 0).numpy() * 0.5 + 0.5)  # Denormalize for display
#     plt.title('Image')
    
#     # Display landmarks
#     plt.subplot(1, 3, 2)
#     plt.imshow(images[0].permute(1, 2, 0).numpy() * 0.5 + 0.5)  # Show the same image
#     landmarks_2d = landmarks[0][:, :2]  # Extract only x and y coordinates
#     # Plot landmarks
#     plt.scatter(landmarks_2d[:, 0].numpy() * images[0].shape[2],  # Scale to image dimensions
#                 landmarks_2d[:, 1].numpy() * images[0].shape[1],  # Scale to image dimensions
#                 s=10, c='r')  # Plot landmarks
#     plt.title('Landmarks')
    
#     # Display depth map
#     plt.subplot(1, 3, 3)
#     plt.imshow(depth_maps[0].numpy().squeeze(), cmap='gray')  # Remove channel dimension for display
#     plt.title('Depth Map')
    
#     plt.show()
#     break  # Show just one batch
