import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def get_landmarks(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    
    # Convert the image to RGB (required by MediaPipe)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image to find face landmarks
    results = face_mesh.process(rgb_image)
    
    # If landmarks are detected, extract them
    if results.multi_face_landmarks:
        # Get the first face's landmarks
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Convert the landmarks to a NumPy array
        landmarks_array = np.array([(lm.x, lm.y, lm.z) for lm in landmarks])
        
        return landmarks_array
    else:
        return None

def save_landmarks(landmarks, output_path):
    # Save the landmarks array to a .npy file
    np.save(output_path, landmarks)

# Directory containing your images
input_dir = "images"
output_dir = "landmarks"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop over all the images in the directory
for image_name in os.listdir(input_dir):
    image_path = os.path.join(input_dir, image_name)
    
    # Get landmarks for the image
    landmarks = get_landmarks(image_path)
    
    if landmarks is not None:
        # Create a filename for the output .npy file
        output_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + ".npy")
        
        # Save the landmarks
        save_landmarks(landmarks, output_path)
        print(f"Saved landmarks for {image_name} to {output_path}")
    else:
        print(f"No face detected in {image_name}")

# Release the resources used by MediaPipe
face_mesh.close()