import cv2
import mediapipe as mp
import numpy as np
import os
import OpenEXR
import Imath

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

# Paths for input images and output depth maps
input_image_folder = 'train_data/images/'
output_depth_map_folder = 'train_data/only_face_depth_maps/'

# Desired output dimensions
output_height = 640
output_width = 360

# Ensure the output folder exists
os.makedirs(output_depth_map_folder, exist_ok=True)

# Process each image in the input folder
for image_filename in os.listdir(input_image_folder):
    if image_filename.endswith('.jpg'):
        # Get the corresponding image file name
        image_name = os.path.splitext(image_filename)[0]
        input_image_path = os.path.join(input_image_folder, image_filename)
        
        # Load the image
        image = cv2.imread(input_image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

                # Create a new depth map with a constant depth of 100
                new_depth_map = np.full((image.shape[0], image.shape[1]), 100, dtype=np.float32)

                # Set the face area in the depth map to 25
                face_depth = np.full((y_max - y_min, x_max - x_min), 25, dtype=np.float32)
                new_depth_map[y_min:y_max, x_min:x_max] = face_depth

                # Resize the new depth map to 360x640
                new_depth_map_resized = cv2.resize(new_depth_map, (output_width, output_height), interpolation=cv2.INTER_LINEAR)

                # Save the new depth map with face depth area filled as EXR
                output_new_depth_map_path = os.path.join(output_depth_map_folder, f'{image_name}.exr')
                save_exr(output_new_depth_map_path, new_depth_map_resized)

                print(f"New depth map saved to {output_new_depth_map_path}")

# Release the face mesh
face_mesh.close()
