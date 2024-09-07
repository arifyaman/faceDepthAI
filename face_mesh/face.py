import cv2
import numpy as np
import mediapipe as mp
import trimesh
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
    """Create a depth map from the mesh that matches the dimensions of the original image."""
    vertices = mesh.vertices
    faces = mesh.faces
    image_height, image_width = image_dims

    # Initialize depth image with infinite values
    depth_image = np.full((image_height, image_width), np.inf)

    x_vals = vertices[:, 0]
    y_vals = vertices[:, 1]
    z_vals = vertices[:, 2]

    # Adjust squeeze factors to align with the image dimensions
    x_squeze_factor = 1.8
    y_squeze_factor = 3.5

    # Interpolate mesh coordinates to image coordinates
    x_img = np.interp(x_vals, (x_vals.min() - x_squeze_factor, x_vals.max() + x_squeze_factor), (0, image_width - 1)).astype(int)
    y_img = np.interp(y_vals, (y_vals.min() - y_squeze_factor - 1, y_vals.max() + y_squeze_factor), (0, image_height - 1)).astype(int)

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

    return depth_image

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

def main():
    # Load and process the image
    image = cv2.imread('image_497.jpg')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extract landmarks
    landmarks = extract_landmarks(image_rgb)
    np.save('extracted_landmarks.npy', landmarks)

    # Apply landmarks to mesh
    mesh = apply_landmarks_to_mesh(landmarks, 'face_model_with_iris.obj')
    mesh.export('updated_face_mesh_478.obj')
    print("Mesh exported successfully to 'updated_face_mesh_478.obj'")

    # Create depth map
    depth_map = create_depth_map(mesh, image.shape[:2])

    # Display images
    display_images(image_rgb, depth_map)

if __name__ == "__main__":
    main()
