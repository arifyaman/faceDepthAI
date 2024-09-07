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
input_depth_map_folder = 'train_data/depth_maps/'
output_depth_map_folder = 'train_data/only_face_depth_maps/'
output_landmarks_folder = 'train_data/landmarks/'

# Ensure the output folders exist
os.makedirs(output_depth_map_folder, exist_ok=True)
os.makedirs(output_landmarks_folder, exist_ok=True)

# Process each image in the input folder
for image_filename in os.listdir(input_image_folder):
    if image_filename.endswith('.jpg'):
        # Get the base name of the image file
        image_name = os.path.splitext(image_filename)[0]
        
        # Paths for the image, depth map, and landmarks
        input_image_path = os.path.join(input_image_folder, image_filename)
        depth_map_path = os.path.join(input_depth_map_folder, f'{image_name}.exr')
        output_depth_map_path = os.path.join(output_depth_map_folder, f'{image_name}.exr')
        output_landmarks_path = os.path.join(output_landmarks_folder, f'{image_name}.npy')
        
        if not os.path.exists(depth_map_path):
            print(f"Depth map for {image_filename} not found. Skipping.")
            continue
        
        # Load the image
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
                # Create a list of landmarks coordinates (normalized values)
                points = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in face_landmarks.landmark]
                int_array_points = np.array([[lm.x * 85, lm.y * 85, lm.z * 10] for lm in face_landmarks.landmark], dtype=np.float32)
                
                # Convert list of points to a NumPy array
                landmarks_array = np.array(points)

                # Save landmarks as .npy file
                np.save(output_landmarks_path, int_array_points)

                print(f"Landmarks saved to {output_landmarks_path}")

                # Create a bounding box from the face landmarks
                points_2d = np.array([(p['x'], p['y']) for p in points], dtype=np.float32)
                x_min, y_min = np.min(points_2d, axis=0)
                x_max, y_max = np.max(points_2d, axis=0)

                # Expand the bounding box slightly for a margin (optional)
                margin = 0  # You can adjust the margin value
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(1, x_max + margin)
                y_max = min(1, y_max + margin)

                # Scale the bounding box coordinates to match the depth map's resolution
                x_scale = depth_map.shape[1] / image.shape[1]
                y_scale = depth_map.shape[0] / image.shape[0]
                x_min_depth = int(x_min * x_scale * image.shape[1])
                y_min_depth = int(y_min * y_scale * image.shape[0])
                x_max_depth = int(x_max * x_scale * image.shape[1])
                y_max_depth = int(y_max * y_scale * image.shape[0])

                # Ensure coordinates are within bounds
                x_min_depth = max(0, x_min_depth)
                y_min_depth = max(0, y_min_depth)
                x_max_depth = min(depth_map.shape[1], x_max_depth)
                y_max_depth = min(depth_map.shape[0], y_max_depth)

                # Crop the depth map based on the scaled bounding box
                depth_map_face = depth_map[y_min_depth:y_max_depth, x_min_depth:x_max_depth]

                # Create a writable copy of the cropped depth map
                depth_map_face_copy = np.copy(depth_map_face)

                # Set all depth values of the cropped face to 100 where the value is 0
                depth_map_face_copy[depth_map_face_copy == 0] = 100

                # Resize the cropped depth map to 85x85
                depth_map_face_resized = cv2.resize(depth_map_face_copy, (85, 85), interpolation=cv2.INTER_LINEAR)

                # Save the resized face depth map as EXR
                save_exr(output_depth_map_path, depth_map_face_resized)

                print(f"Resized depth map saved to {output_depth_map_path}")

# Release the face mesh
face_mesh.close()
