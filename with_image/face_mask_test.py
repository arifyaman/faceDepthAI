import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize Mediapipe face detection and landmark modules
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Define colors for each facial region
colors = {
    'lips': (255, 0, 255),       # Magenta
    'left_eye': (0, 255, 0),     # Green
    'left_iris': (0, 200, 0),    # Dark Green
    'left_eyebrow': (255, 0, 0), # Red
    'right_eye': (0, 255, 255),  # Cyan
    'right_eyebrow': (255, 0, 0),# Red
    'right_iris': (0, 200, 200), # Dark Cyan
    'face_oval': (200, 200, 200), # Light Gray
    'nose': (0, 0, 255)          # Blue
}

def detect_landmarks(image):
    """Detect face landmarks in the given image using Mediapipe."""
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5, refine_landmarks=True) as face_mesh:
        
        # Convert the image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        
        landmarks_list = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Scale normalized landmarks to image dimensions
                landmarks = [(int(lmk.x * image.shape[1]), int(lmk.y * image.shape[0])) for lmk in face_landmarks.landmark]
                landmarks_list.append(landmarks)
                
        return landmarks_list

def create_colored_masks(image, landmarks):
    """Create colored masks for facial regions based on landmarks."""
    masks = {key: np.zeros_like(image) for key in colors}
    
    # Define facial feature classes
    classes = [
        mp_face_mesh.FACEMESH_LIPS,
        mp_face_mesh.FACEMESH_LEFT_EYE,
        mp_face_mesh.FACEMESH_LEFT_IRIS,
        mp_face_mesh.FACEMESH_LEFT_EYEBROW,
        mp_face_mesh.FACEMESH_RIGHT_EYE,
        mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
        mp_face_mesh.FACEMESH_RIGHT_IRIS,
        mp_face_mesh.FACEMESH_FACE_OVAL,
        mp_face_mesh.FACEMESH_NOSE,
    ]
    
    # Define class descriptions
    class_descriptions = {
        'lips': mp_face_mesh.FACEMESH_LIPS,
        'left_eye': mp_face_mesh.FACEMESH_LEFT_EYE,
        'left_iris': mp_face_mesh.FACEMESH_LEFT_IRIS,
        'left_eyebrow': mp_face_mesh.FACEMESH_LEFT_EYEBROW,
        'right_eye': mp_face_mesh.FACEMESH_RIGHT_EYE,
        'right_eyebrow': mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
        'right_iris': mp_face_mesh.FACEMESH_RIGHT_IRIS,
        'face_oval': mp_face_mesh.FACEMESH_FACE_OVAL,
        'nose': mp_face_mesh.FACEMESH_NOSE,
    }

    for face_landmarks in landmarks:
        # Convert landmarks to numpy array
        points = np.array(face_landmarks, np.int32)
        
        for region, connections in class_descriptions.items():
            for connection in connections:
                start_idx, end_idx = connection
                start_point = tuple(points[start_idx])
                end_point = tuple(points[end_idx])
                
                # Draw lines between connected landmarks for visualization
                cv2.line(masks[region], start_point, end_point, colors[region], 2)
                
            # Fill convex polygons if there are enough points to form a region
            if region in ['left_eye', 'right_eye', 'left_eyebrow', 'right_eyebrow', 'face_oval', 'nose']:
                # Find all points in the region and draw a convex polygon
                polygon_points = [points[i] for i in range(len(points)) if i in [c[0] for c in connections]]
                if len(polygon_points) > 2:
                    cv2.fillConvexPoly(masks[region], np.array(polygon_points, np.int32), colors[region])
    
    return masks

def apply_masks(image, masks):
    """Apply colored masks to the image."""
    combined_mask = np.zeros_like(image)
    
    for mask in masks.values():
        combined_mask = cv2.addWeighted(combined_mask, 1.0, mask, 1.0, 0)
    
    masked_image = cv2.addWeighted(image, 0.3, combined_mask, 0.7, 0)
    
    return masked_image

def show_images(original, masked):
    """Display the original and masked images."""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Masked Image")
    plt.imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    # Path to the image
    image_path = "images/image_17.jpg"
    
    # Read the image
    image = cv2.imread(image_path)
    
    # Detect landmarks
    landmarks = detect_landmarks(image)
    
    if landmarks:
        # Create masks with colors
        masks = create_colored_masks(image, landmarks)
        
        # Apply masks to the image
        masked_image = apply_masks(image, masks)
        
        # Display images
        show_images(image, masked_image)
    else:
        print("No landmarks detected. Please check the image or adjust detection parameters.")
