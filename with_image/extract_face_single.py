import cv2
import mediapipe as mp
import numpy as np
import os
import OpenEXR
import Imath
import matplotlib.pyplot as plt

# Function to save EXR
def save_exr(output_path, depth_map):
    # Define EXR channel format
    header = OpenEXR.Header(depth_map.shape[1], depth_map.shape[0])
    header['channels'] = {'Y': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    
    # Create an EXR file
    exr_file = OpenEXR.OutputFile(output_path, header)
    
    # Write only the Y (depth) channel
    exr_file.writePixels({'Y': depth_map.astype(np.float32).tobytes()})
    exr_file.close()

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Load the image and depth map
input_image_path = 'images/image_2583.jpg'
depth_map_path = 'depth_maps/image_2583.exr'
output_image_folder = 'only_face_out'

image = cv2.imread(input_image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load the depth map (assuming Y channel contains depth, single-channel EXR)
exr_file = OpenEXR.InputFile(depth_map_path)
header = exr_file.header()
dw = header['dataWindow']
width = dw.max.x - dw.min.x + 1
height = dw.max.y - dw.min.y + 1
depth_map = np.frombuffer(exr_file.channel('Y', Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32).reshape((height, width))

# Process the image and extract landmarks
results = face_mesh.process(image_rgb)

if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # Create a list of landmarks coordinates
        points = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            points.append((x, y))

        # Create a bounding box from the face landmarks
        points = np.array(points, dtype=np.int32)
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)

        # Expand the bounding box slightly for a margin (optional)
        margin = 0  # You can adjust the margin value
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(image.shape[1], x_max + margin)
        y_max = min(image.shape[0], y_max + margin)

        # Scale the bounding box coordinates to match the depth map's resolution
        x_scale = depth_map.shape[1] / image.shape[1]
        y_scale = depth_map.shape[0] / image.shape[0]
        x_min_depth = int(x_min * x_scale)
        y_min_depth = int(y_min * y_scale)
        x_max_depth = int(x_max * x_scale)
        y_max_depth = int(y_max * y_scale)

        # Crop the depth map based on the scaled bounding box
        depth_map_face = depth_map[y_min_depth:y_max_depth, x_min_depth:x_max_depth]

        # Resize the cropped depth map to 200x200
        depth_map_face_resized = cv2.resize(depth_map_face, (85, 85), interpolation=cv2.INTER_LINEAR)

        # Save the resized depth map as EXR
        output_depth_map_resized_path = os.path.join(output_image_folder, 'face_only_depth_85x85.exr')
        save_exr(output_depth_map_resized_path, depth_map_face_resized)

        print(f"Resized depth map saved to {output_depth_map_resized_path}")

        # Visualization: Display the original depth map, cropped face region, and resized depth map
        plt.figure(figsize=(15, 5))

        # Show the original depth map
        plt.subplot(1, 3, 1)
        plt.imshow(depth_map, cmap='viridis')
        plt.title('Original Depth Map')
        plt.axis('off')

        # Show the cropped face region
        plt.subplot(1, 3, 2)
        plt.imshow(depth_map_face, cmap='viridis')
        plt.title('Cropped Face Depth Map')
        plt.axis('off')

        # Show the resized 200x200 depth map
        plt.subplot(1, 3, 3)
        plt.imshow(depth_map_face_resized, cmap='viridis')
        plt.title('Resized 200x200 Depth Map')
        plt.axis('off')

        # Display all three images
        plt.show()

# Release the face mesh
face_mesh.close()
