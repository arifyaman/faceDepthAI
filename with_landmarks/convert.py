import cv2
import mediapipe as mp
import numpy as np
import os
import torch
import numpy as np
from model import SimpleLandmarkOnlyCNN  # Ensure this matches the class name of your model
import OpenEXR
import Imath
import os
import matplotlib.pyplot as plt  # For displaying the depth map

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

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model_path = 'model_epoch_200.pth'  # Adjust the path if needed
model = SimpleLandmarkOnlyCNN()  # Use the class for landmarks-only model
model.load_state_dict(torch.load(model_path, map_location=device))  # Load model to the device
model.to(device)  # Move the model to the GPU if available
model.eval()

# Function to save depth map as EXR
def save_exr_depth_map(depth_map, exr_path):
    height, width = depth_map.shape
    header = OpenEXR.Header(width, height)
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels'] = {'Y': half_chan}
    
    exr_file = OpenEXR.OutputFile(exr_path, header)
    
    # Convert the depth map to bytes
    depth_bytes = depth_map.astype(np.float32).tobytes()
    
    # Write depth map to EXR
    exr_file.writePixels({'Y': depth_bytes})
    exr_file.close()

# Directories
landmarks_dir = "landmarks"
output_dir = "depth_maps_output"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop over all the landmarks files in the directory
for landmarks_name in os.listdir(landmarks_dir):
    landmarks_path = os.path.join(landmarks_dir, landmarks_name)
    
    # Load the landmarks
    landmarks = np.load(landmarks_path)  # Load landmarks as a numpy array
    landmarks = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    # Move the landmarks tensor to the GPU if available
    landmarks = landmarks.to(device)

    # Run the model
    with torch.no_grad():
        output_depth_map = model(landmarks)  # Pass only landmarks

    # Convert the output to a numpy array
    output_depth_map_np = output_depth_map.squeeze().cpu().numpy()  # Remove batch and channel dimensions and move to CPU

    # Save the depth map as an EXR file
    output_exr_path = os.path.join(output_dir, os.path.splitext(landmarks_name)[0] + ".exr")
    save_exr_depth_map(output_depth_map_np, output_exr_path)


    print(f"Depth map saved to {output_exr_path}")