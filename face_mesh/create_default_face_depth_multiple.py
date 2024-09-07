import os
import cv2
import numpy as np
import mediapipe as mp
import trimesh
import matplotlib.pyplot as plt
import OpenEXR
import Imath
import concurrent.futures

def save_exr(output_path, depth_map):
    # Define EXR channel format
    header = OpenEXR.Header(depth_map.shape[1], depth_map.shape[0])
    header['channels'] = {'Y': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    
    # Create an EXR file
    exr_file = OpenEXR.OutputFile(output_path, header)
    
    # Write only the Y (depth) channel
    exr_file.writePixels({'Y': depth_map.astype(np.float32).tobytes()})
    exr_file.close()

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

def create_depth_map(mesh, image_dims, bbox, scale_factor=1.0, base_position=20.0):
    """Create a depth map from the mesh and resize it to the bounding box dimensions."""
    vertices = mesh.vertices
    faces = mesh.faces
    image_height, image_width = image_dims

    # Initialize depth image with infinite values
    depth_image = np.full((image_height, image_width), np.inf)

    x_vals = vertices[:, 0]
    y_vals = vertices[:, 1]
    z_vals = vertices[:, 2]

    # Interpolate mesh coordinates to image coordinates
    x_img = np.interp(x_vals, (x_vals.min(), x_vals.max()), (0, image_width - 1)).astype(int)
    y_img = np.interp(y_vals, (y_vals.min() - 1, y_vals.max()), (0, image_height - 1)).astype(int)

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

    # Replace infinite values with the minimum depth
    depth_image[np.isinf(depth_image)] = 0

    # Normalize depth map to the range [1, 100]
    depth_image = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image) + 1e-6) * 99 + 1

    # Scale with scale factor
    depth_image *= scale_factor

    # add the base position to all depth map values
    depth_image += base_position

    # Extract bounding box coordinates
    x_min, y_min, x_max, y_max = bbox

    # Ensure bounding box is within image dimensions
    x_min, x_max = int(max(0, x_min)), int(min(image_width - 1, x_max))
    y_min, y_max = int(max(0, y_min)), int(min(image_height - 1, y_max))

    # Calculate bounding box dimensions
    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1

    depth_image[np.less(depth_image,base_position+2)] = 100

    # Resize depth image to the bounding box dimensions
    resized_depth_image = cv2.resize(depth_image, (bbox_width, bbox_height), interpolation=cv2.INTER_LINEAR)
    
    return resized_depth_image

def create_final_depth_map(resized_depth_map, bbox, original_dims, target_dims=(360, 640)):
    """Create a new depth map with the resized depth map inserted into the bounding box, then resize the final map."""
    # Initialize the depth map with a constant depth value of 100
    original_height, original_width = original_dims
    depth_map = np.full((original_height, original_width), 100, dtype=np.float32)

    # Extract bounding box coordinates
    x_min, y_min, x_max, y_max = bbox

    # Calculate target bounding box coordinates for the depth map insertion
    x_min_target = int(max(0, x_min))
    y_min_target = int(max(0, y_min))
    x_max_target = int(min(original_width - 1, x_min_target + resized_depth_map.shape[1] - 1))
    y_max_target = int(min(original_height - 1, y_min_target + resized_depth_map.shape[0] - 1))

    # Calculate the width and height of the adjusted resized depth map
    adjusted_width = x_max_target - x_min_target + 1
    adjusted_height = y_max_target - y_min_target + 1

    # Ensure that resized depth map fits into the calculated target area
    adjusted_resized_depth_map = resized_depth_map[
        :adjusted_height, :adjusted_width
    ]

    # Insert the resized depth map into the final depth map
    depth_map[y_min_target:y_max_target + 1, x_min_target:x_max_target + 1] = adjusted_resized_depth_map

    # Resize the depth map to the target dimensions
    final_depth_map = cv2.resize(depth_map, target_dims, interpolation=cv2.INTER_LINEAR)

    return final_depth_map

def display_images(original_image, depth_image):
    """Display the original image and depth map side by side."""
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(depth_image, cmap='plasma')
    plt.colorbar(label='Depth')
    plt.title('Depth Map')
    plt.axis('off')

    plt.show()

def translate_camera(mesh, translation_vector):
    """
    Translates the vertices of the mesh according to the given translation vector.
    
    Parameters:
    - mesh: The 3D mesh object (trimesh.Trimesh).
    - translation_vector: A 3D vector (x, y, z) representing the translation.
    
    Returns:
    - Translated mesh (with updated vertices).
    """
    translated_vertices = mesh.vertices + translation_vector
    mesh.vertices = translated_vertices
    return mesh

def rotate_camera(mesh, rotation_angles):
    """
    Rotates the vertices of the mesh according to the given rotation angles.
    
    Parameters:
    - mesh: The 3D mesh object (trimesh.Trimesh).
    - rotation_angles: A 3D vector (rx, ry, rz) representing the rotation angles in radians
                       around the X, Y, and Z axes.
    
    Returns:
    - Rotated mesh (with updated vertices).
    """
    rx, ry, rz = rotation_angles

    # Rotation matrices for X, Y, and Z axes
    rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(rx), -np.sin(rx)],
                           [0, np.sin(rx), np.cos(rx)]])
    
    rotation_y = np.array([[np.cos(ry), 0, np.sin(ry)],
                           [0, 1, 0],
                           [-np.sin(ry), 0, np.cos(ry)]])
    
    rotation_z = np.array([[np.cos(rz), -np.sin(rz), 0],
                           [np.sin(rz), np.cos(rz), 0],
                           [0, 0, 1]])

    # Combined rotation matrix
    rotation_matrix = rotation_z @ rotation_y @ rotation_x

    # Rotate vertices
    rotated_vertices = np.dot(mesh.vertices, rotation_matrix.T)
    mesh.vertices = rotated_vertices

    return mesh

def process_image(image_file, input_folder, output_folder):
    """Process a single image file and save the depth map."""
    # Construct full file paths
    image_path = os.path.join(input_folder, image_file)
    output_depth_map_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + '.exr')

    # Load and process the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extract landmarks
    landmarks = extract_landmarks(image_rgb)

    # Calculate bounding box
    x_min, y_min = np.min(landmarks[:, :2], axis=0)
    x_max, y_max = np.max(landmarks[:, :2], axis=0)
    bbox = (x_min, y_min, x_max, y_max)

    # Apply landmarks to mesh
    mesh = apply_landmarks_to_mesh(landmarks, 'face_model_with_iris.obj')

    # Translate the camera (translate mesh vertices)
    tranlation_vector = (0, 0, 0)
    mesh = translate_camera(mesh, tranlation_vector)

    # Rotate the camera (rotate the object)
    rotation_angles = np.radians([-15, 0, 0])  # Rotate by 10° around X, 15° around Y
    mesh = rotate_camera(mesh, rotation_angles)
    
    # Create and resize depth map
    resized_depth_map = create_depth_map(mesh, image.shape[:2], bbox, 0.2)

    # Create final depth map
    final_depth_map = create_final_depth_map(resized_depth_map, bbox, image.shape[:2], target_dims=(360, 640))

    # Save the final depth map
    save_exr(output_depth_map_path, final_depth_map)

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