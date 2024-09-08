import cv2
import numpy as np
import mediapipe as mp
import trimesh
import matplotlib.pyplot as plt
import OpenEXR
import Imath

class DepthMapProcessor:

    def save_exr(self, output_path, depth_map):
        # Define EXR channel format
        header = OpenEXR.Header(depth_map.shape[1], depth_map.shape[0])
        header['channels'] = {'Y': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
        
        # Create an EXR file
        exr_file = OpenEXR.OutputFile(output_path, header)
        
        # Write only the Y (depth) channel
        exr_file.writePixels({'Y': depth_map.astype(np.float32).tobytes()})
        exr_file.close()

    def extract_landmarks(self, image):
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

    def apply_landmarks_to_mesh(self, landmarks, mesh_file):
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

    def create_depth_map(self, mesh, image_dims, bbox, scale_factor=1.0, base_position=20.0):
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

    def create_final_depth_map(self, resized_depth_map, bbox, original_dims, target_dims=(360, 640), scale_factors=(1.0, 1.0), margin=(0, 0)):
        """Create a new depth map with the resized depth map (scaled and translated) inserted into the bounding box, then resize the final map."""
        # Initialize the depth map with a constant depth value of 100
        original_height, original_width = original_dims
        depth_map = np.full((original_height, original_width), 100, dtype=np.float32)

        # Extract bounding box coordinates
        x_min, y_min, x_max, y_max = bbox

        # Calculate the center of the bounding box
        bbox_center_x = (x_min + x_max) / 2
        bbox_center_y = (y_min + y_max) / 2

        # Scale factors for width and height
        scale_x, scale_y = scale_factors

        # Margin vector for translation (x_margin, y_margin)
        x_margin, y_margin = margin

        # Calculate the new size of the resized depth map after scaling
        new_height = int(resized_depth_map.shape[0] * scale_y)
        new_width = int(resized_depth_map.shape[1] * scale_x)
        
        # Resize the resized_depth_map to the new dimensions
        scaled_depth_map = cv2.resize(resized_depth_map, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Calculate new bounding box coordinates based on the scaled depth map and margin
        x_min_target = int(max(0, bbox_center_x - new_width / 2 + x_margin))
        y_min_target = int(max(0, bbox_center_y - new_height / 2 + y_margin))
        x_max_target = int(min(original_width - 1, x_min_target + new_width - 1))
        y_max_target = int(min(original_height - 1, y_min_target + new_height - 1))

        # Ensure the scaled depth map fits into the calculated target area
        adjusted_width = x_max_target - x_min_target + 1
        adjusted_height = y_max_target - y_min_target + 1
        scaled_depth_map = scaled_depth_map[:adjusted_height, :adjusted_width]

        # Insert the scaled depth map into the final depth map
        depth_map[y_min_target:y_max_target + 1, x_min_target:x_max_target + 1] = scaled_depth_map

        # Resize the depth map to the target dimensions
        final_depth_map = cv2.resize(depth_map, target_dims, interpolation=cv2.INTER_LINEAR)

        return final_depth_map

    def display_images(self, original_image, depth_image):
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

    def translate_camera(self, mesh, translation_vector):
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

    def rotate_camera(self, mesh, rotation_angles):
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