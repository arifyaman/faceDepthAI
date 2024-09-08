import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from depth_map_processor import DepthMapProcessor

processor = DepthMapProcessor()

# Parameters
input_image_path = 'images/0001.jpg'
output_depth_map_folder = ''
output_depth_map_filename = 'depth_map_sample.exr'
face_model_path = 'face_model_with_iris.obj'

#### IMPORTANT #### Fine tune these paramters for one of your image to see mask matches with your face

# Parameters for DepthMapProcessor
depth_map_params = {
    'scale_factor': 0.2,                   # Scale factor for depth map creation
    'target_dims': (360, 640),             # Target dimensions for the final depth map
    'scale_factors': (1, 1.15),            # Scale factors for width and height only for face area
    'margin': (0, -20)                     # Margin for translating the face area
}

# Camera translation and rotation parameters
camera_params = {
    'translation_vector': (0, 0, 0),       # Translation vector for the camera
    'rotation_angles': np.radians([-15, 0, 0])  # Rotation angles (in radians) for the camera
}

# Initialize processor
processor = DepthMapProcessor()

# Load and process the image
image = cv2.imread(input_image_path)
output_depth_map_path = os.path.join(output_depth_map_folder, output_depth_map_filename)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Extract landmarks
landmarks = processor.extract_landmarks(image_rgb)

# Calculate bounding box
x_min, y_min = np.min(landmarks[:, :2], axis=0)
x_max, y_max = np.max(landmarks[:, :2], axis=0)
bbox = (x_min, y_min, x_max, y_max)
print(f"Bounding box: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")

# Apply landmarks to mesh
mesh = processor.apply_landmarks_to_mesh(landmarks, face_model_path)

# Translate the camera (translate mesh vertices)
mesh = processor.translate_camera(mesh, camera_params['translation_vector'])

# Rotate the camera (rotate the object)
mesh = processor.rotate_camera(mesh, camera_params['rotation_angles'])

# Create and resize depth map
resized_depth_map = processor.create_depth_map(
    mesh, 
    image.shape[:2], 
    bbox, 
    depth_map_params['scale_factor']
)

# Create final depth map
final_depth_map = processor.create_final_depth_map(
    resized_depth_map, 
    bbox, 
    image.shape[:2], 
    target_dims=depth_map_params['target_dims'], 
    scale_factors=depth_map_params['scale_factors'], 
    margin=depth_map_params['margin']
)

# Save depth map
processor.save_exr(output_depth_map_path, final_depth_map)

# Normalize depth map for overlay display (scale to [0, 255])
normalized_depth_map = cv2.normalize(final_depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Apply a colormap for visualization
depth_map_colored = cv2.applyColorMap(normalized_depth_map, cv2.COLORMAP_PLASMA)

# Resize depth_map_colored to match image_rgb size
depth_map_colored_resized = cv2.resize(depth_map_colored, (image_rgb.shape[1], image_rgb.shape[0]))

# Blend the original image with the depth map for overlay
blended_image = cv2.addWeighted(image_rgb, 0.6, depth_map_colored_resized, 0.4, 0)

# Display the images
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(final_depth_map, cmap='plasma')
plt.colorbar(label='Depth')
plt.title('Colored Depth Map')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(blended_image)
plt.title('Overlay of Original Image and Depth Map')
plt.axis('off')

plt.show()