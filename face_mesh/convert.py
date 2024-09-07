import cv2
import numpy as np
import mediapipe as mp
import trimesh
import os
import OpenEXR
import Imath
import matplotlib.pyplot as plt

def extract_landmarks(image):
    """Extract 478 face landmarks from an image using MediaPipe."""
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    results = face_mesh.process(image)

    if results.multi_face_landmarks:
        landmarks = []
        for landmark in results.multi_face_landmarks[0].landmark:
            h, w, _ = image.shape
            landmarks.append([landmark.x * w, landmark.y * h, landmark.z * w])
        landmarks = np.array(landmarks)

        if landmarks.shape[0] != 478:
            raise ValueError(f"Expected 478 landmarks, but got {landmarks.shape[0]}.")
        
        return landmarks
    else:
        raise ValueError("Could not extract landmarks from the image.")

def apply_landmarks_to_mesh(landmarks, mesh_file):
    """Apply extracted landmarks to the default face mesh and return the updated mesh."""
    default_mesh = trimesh.load(mesh_file)

    if default_mesh.vertices.shape[0] != 478:
        raise ValueError(f"The default mesh must have 478 vertices. Found {default_mesh.vertices.shape[0]} instead.")

    # Scale and align landmarks to fit the default mesh
    scale_factor = np.max(np.abs(default_mesh.vertices)) / np.max(np.abs(landmarks))
    landmarks *= scale_factor * 2
    landmarks += default_mesh.vertices.mean(axis=0) - landmarks.mean(axis=0)

    default_mesh.vertices = landmarks
    return default_mesh

def create_depth_map(mesh, image_dims):
    """Create a depth map from the mesh that matches the dimensions of the original image and resize it."""
    vertices = mesh.vertices
    faces = mesh.faces
    image_height, image_width = image_dims

    # Initialize depth image with infinite values
    depth_image = np.full((image_height, image_width), np.inf)

    x_vals = vertices[:, 0]
    y_vals = vertices[:, 1]
    z_vals = vertices[:, 2]

    # Adjust squeeze factors to align with the image dimensions
    x_squeeze_factor = 1.8
    y_squeeze_factor = 3.5

    # Interpolate mesh coordinates to image coordinates
    x_img = np.interp(x_vals, (x_vals.min() - x_squeeze_factor, x_vals.max() + x_squeeze_factor), (0, image_width - 1)).astype(int)
    y_img = np.interp(y_vals, (y_vals.min() - y_squeeze_factor - 1, y_vals.max() + y_squeeze_factor), (0, image_height - 1)).astype(int)

    # Rasterization of the depth values into the depth map
    for face in faces:
        x_face = x_img[face]
        y_face = y_img[face]
        z_face = z_vals[face]

        # Sort vertices by their y-coordinate
        indices = np.argsort(y_face)
        x_face = x_face[indices]
        y_face = y_face[indices]
        z_face = z_face[indices]

        # Fill the depth map with interpolated depth values
        for i in range(y_face[0], y_face[2] + 1):
            if i < y_face[1]:
                t = (i - y_face[0]) / (y_face[1] - y_face[0] + 1e-6)
                x_start = int(x_face[0] + t * (x_face[1] - x_face[0]))
                z_start = z_face[0] + t * (z_face[1] - z_face[0])
            else:
                t = (i - y_face[1]) / (y_face[2] - y_face[1] + 1e-6)
                x_start = int(x_face[1] + t * (x_face[2] - x_face[1]))
                z_start = z_face[1] + t * (z_face[2] - z_face[1])

            t = (i - y_face[0]) / (y_face[2] - y_face[0] + 1e-6)
            x_end = int(x_face[0] + t * (x_face[2] - x_face[0]))
            z_end = z_face[0] + t * (z_face[2] - z_face[0])

            if x_start > x_end:
                x_start, x_end = x_end, x_start
                z_start, z_end = z_end, z_start

            for j in range(x_start, x_end + 1):
                t = (j - x_start) / (x_end - x_start + 1e-6)
                depth = z_start + t * (z_end - z_start)
                if depth < depth_image[i, j]:
                    depth_image[i, j] = depth

    depth_image = 1 - depth_image
    # Replace infinite values with the minimum depth
    depth_image[np.isinf(depth_image)] = 0

    # Normalize depth map to the range [1, 100]
    depth_image = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image) + 1e-6) * 100

    # Resize depth map to 360x640
    target_size = (360, 640)
    depth_image_resized = cv2.resize(depth_image, target_size, interpolation=cv2.INTER_LINEAR)

    return depth_image_resized

def save_depth_map_as_exr(depth_image, filename):
    """Save the depth map as an .exr file."""
    # Create an OpenEXR file
    header = OpenEXR.Header(depth_image.shape[1], depth_image.shape[0])
    exr_file = OpenEXR.OutputFile(filename, header)

    # Convert depth image to EXR format (Y channel)
    depth_image = depth_image.astype(np.float32)  # Ensure the type is float32
    
    # Write depth image to EXR file
    exr_file.writePixels({'Y': depth_image.tobytes()})
    exr_file.close()

def process_image(image_path, output_folder, mesh_file):
    """Process a single image to generate and save a depth map."""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extract landmarks
    landmarks = extract_landmarks(image_rgb)
    
    # Apply landmarks to mesh
    mesh = apply_landmarks_to_mesh(landmarks, mesh_file)

    # Create depth map
    depth_map = create_depth_map(mesh, image.shape[:2])

    # Save depth map as .exr file
    image_name = os.path.basename(image_path)
    depth_map_filename = os.path.join(output_folder, image_name.replace('.jpg', '.exr'))
    save_depth_map_as_exr(depth_map, depth_map_filename)
    print(f"Depth map saved to '{depth_map_filename}'")
    # plt.figure(figsize=(12, 6))

    # plt.subplot(1, 2, 1)
    # plt.imshow(image_rgb)
    # plt.title('Original Image')
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.imshow(depth_map, cmap='plasma')
    # plt.colorbar(label='Depth')
    # plt.title('Depth Map')
    # plt.axis('off')

    # plt.show()

def main():
    # Define input and output folders
    input_folder = 'images'
    output_folder = 'depth_maps'
    mesh_file = 'face_model_with_iris.obj'

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):
            image_path = os.path.join(input_folder, filename)
            process_image(image_path, output_folder, mesh_file)
            # break
if __name__ == "__main__":
    main()
