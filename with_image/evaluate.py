import torch
import cv2
import numpy as np
from model import SimpleCNN
import OpenEXR
import Imath
from torchvision import transforms
import matplotlib.pyplot as plt

def load_checkpoint(model, checkpoint_path):
    """
    Load model state from a checkpoint file.
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model_path = 'model_epoch_5000.pth'
model = SimpleCNN()

# Load model state from checkpoint
model = load_checkpoint(model, model_path)
model.to(device)
model.eval()

# Define the transformation to match the training preprocessing
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

# Load the image
image_path = 'evaluate/images/frame0151.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_transformed = transform(image_rgb).unsqueeze(0)  # Add batch dimension

# Move the image tensor to the GPU if available
image_transformed = image_transformed.to(device)

# Run the model
with torch.no_grad():
    output_depth_map = model(image_transformed)

# Convert the output to a numpy array
output_depth_map_np = output_depth_map.squeeze().cpu().numpy()

# Function to apply conditional smoothing based on depth similarity
def apply_conditional_smoothing(depth_map, threshold=20, kernel_size=5, sigma=10):
    """
    Apply Gaussian smoothing only to areas where neighboring depth values are within a certain threshold.
    """
    smoothed_depth_map = np.copy(depth_map)  # Copy to preserve the original
    
    # Create a mask for where smoothing should be applied
    mask = np.zeros_like(depth_map, dtype=bool)
    
    # Iterate over each pixel
    for i in range(depth_map.shape[0]):
        for j in range(depth_map.shape[1]):
            # Get a local window around the current pixel
            y_start = max(i - kernel_size // 2, 0)
            y_end = min(i + kernel_size // 2 + 1, depth_map.shape[0])
            x_start = max(j - kernel_size // 2, 0)
            x_end = min(j + kernel_size // 2 + 1, depth_map.shape[1])
            
            # Extract the local region
            local_region = depth_map[y_start:y_end, x_start:x_end]
            
            # Check if the maximum difference in this region is below the threshold
            if np.max(local_region) - np.min(local_region) < threshold:
                mask[i, j] = True
    
    # Apply Gaussian smoothing only in the regions identified by the mask
    smoothed_depth_map = cv2.GaussianBlur(depth_map, (kernel_size, kernel_size), sigma)
    
    # Combine smoothed and original depth maps based on the mask
    result = np.where(mask, smoothed_depth_map, depth_map)
    
    return result

# Apply conditional smoothing to the depth map
output_depth_map_cond_smoothed_np = apply_conditional_smoothing(output_depth_map_np, threshold=1500, kernel_size=5, sigma=10)

# Save the smoothed depth map as an EXR file
output_exr_path = 'C:/Users/Xlip/Documents/Unreal Projects/RigTests/Content/MetaHumans/AAFacial/AndroidFootage/DepthFrames/output_depth_map_cond_smoothed.exr'
save_exr_depth_map(output_depth_map_np, output_exr_path)

# Display the original image and the depth map using matplotlib
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Display the original image
ax[0].imshow(image_rgb)
ax[0].set_title('Original Image')
ax[0].axis('off')

# Display the conditionally smoothed depth map
cax = ax[1].imshow(output_depth_map_np, cmap='plasma')
ax[1].set_title('Conditionally Smoothed Depth Map')
ax[1].axis('off')

# Add a colorbar to the depth map plot
fig.colorbar(cax, ax=ax[1], label='Depth')

plt.show()

print(f"Conditionally smoothed depth map saved to {output_exr_path}")
