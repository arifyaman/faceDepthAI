import cv2
import numpy as np
import os
import torch
from model import SimpleCNN  # Ensure this matches the class name of your SimpleCNN model
import OpenEXR
import Imath
import matplotlib.pyplot as plt
from torchvision import transforms

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model_path = 'models/model_epoch_100.pth'  # Adjust the path if needed
model = SimpleCNN()  # Use the class for SimpleCNN model
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))  # Load model to the device
model.to(device)  # Move the model to the GPU if available
model.eval()

# Define the transformation to match the training preprocessing (if applicable)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Function to save depth map as EXR
def save_exr_depth_map(depth_map, exr_path):
    height, width = depth_map.shape
    header = OpenEXR.Header(width, height)
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels'] = {'Y': half_chan}
    
    exr_file = OpenEXR.OutputFile(exr_path, header)
    
    # Convert the depth map to bytes
    depth_bytes = depth_map.astype(np.float32).tobytes()
    
    # Write depth map to EXR
    exr_file.writePixels({'Y': depth_bytes})
    exr_file.close()

# Directories
input_dir = "evaluate/images"  # Directory containing your input images
output_dir = "evaluate/depth_maps_output"  # Directory to save the depth map outputs

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop over all the images in the directory
for image_name in os.listdir(input_dir):
    image_path = os.path.join(input_dir, image_name)
    
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_transformed = transform(image_rgb).unsqueeze(0)  # Add batch dimension
    
    # Move the image tensor to the GPU if available
    image_transformed = image_transformed.to(device)
    
    # Run the model
    with torch.no_grad():
        output_depth_map = model(image_transformed)  # Pass the transformed image

    # Convert the output to a numpy array
    output_depth_map_np = output_depth_map.squeeze().cpu().numpy()  # Remove batch and channel dimensions and move to CPU

    # # Invert the depth map
    # output_depth_map_np = output_depth_map_np.max() - output_depth_map_np
    # output_depth_map_np += 10

    # Save the inverted depth map as an EXR file
    output_exr_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + ".exr")
    save_exr_depth_map(output_depth_map_np, output_exr_path)

    print(f"Depth map saved to {output_exr_path}")
