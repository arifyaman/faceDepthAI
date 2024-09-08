import os
import cv2
import numpy as np
import concurrent.futures
from depth_map_processor import DepthMapProcessor
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Suppresses WARNING logs

# Initialize the DepthMapProcessor
processor = DepthMapProcessor()

# Parameters for DepthMapProcessor
depth_map_params = {
    'scale_factor': 0.2,                   # Scale factor for depth map values probably no need to play around, it is configured for Iphone15
    'target_dims': (360, 640),             # Target dimensions for the final depth map
    'face_scale_factors': (1, 1.15),       # Scale factors for width and height only for face area
    'margin': (0, -20)                     # Margin for translating the face area
}

# Camera translation and rotation parameters
camera_params = {
    'translation_vector': (0, 0, 0),            # Translation vector for the camera
    'rotation_angles': np.radians([-15, 0, 0])  # Rotation angles (in radians) for the camera
}

def process_image(image_file, input_folder, output_folder):
    """Process a single image file and save the depth map."""
    # Construct full file paths
    image_path = os.path.join(input_folder, image_file)
    output_depth_map_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + '.exr')

    # Load and process the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_file}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extract landmarks
    landmarks = processor.extract_landmarks(image_rgb)

    # Calculate bounding box
    x_min, y_min = np.min(landmarks[:, :2], axis=0)
    x_max, y_max = np.max(landmarks[:, :2], axis=0)
    bbox = (x_min, y_min, x_max, y_max)

    # Apply landmarks to mesh
    mesh = processor.apply_landmarks_to_mesh(landmarks, 'face_model_with_iris.obj')

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
        face_scale_factors=depth_map_params['face_scale_factors'], 
        margin=depth_map_params['margin']
    )

    # Save the final depth map
    processor.save_exr(output_depth_map_path, final_depth_map)

    print(f"Saved depth map to: {output_depth_map_path}")

def main():
    input_folder = 'images'  # Update this with your input folder path
    output_folder = 'depth_maps'  # Update this with your output folder path

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map the process_image function to all image files
        futures = [executor.submit(process_image, image_file, input_folder, output_folder) for image_file in image_files]
        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # Retrieve result to check for exceptions
            except Exception as e:
                print(f"An error occurred: {e}")

    print("Processing complete.")

if __name__ == "__main__":
    main()
