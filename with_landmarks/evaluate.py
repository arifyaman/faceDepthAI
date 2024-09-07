import torch
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from model import LandmarkToDepthCNN  # Updated model import

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def get_landmarks(image):
    """ Extract face landmarks from an image using MediaPipe. """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        landmarks_array = np.array([(lm.x * 85, lm.y * 85, lm.z * 10) for lm in landmarks])
        return landmarks_array
    else:
        raise ValueError("No face detected in the image")

def preprocess_landmarks(landmarks_array, landmark_dim=468):
    """ Preprocess landmarks for model input. """
    # Ensure landmarks are reshaped to match the model's expected input shape: [batch_size, landmark_dim, 3]
    landmarks_tensor = torch.tensor(landmarks_array, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    return landmarks_tensor

def display_results(image, depth_map):
    """ Display the image and depth map side by side. """
    depth_map = depth_map.squeeze().cpu().detach().numpy()
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(depth_map, cmap='plasma')
    plt.title('Depth Map Prediction')
    plt.show()

def test(image_path, model_path, landmark_dim=468, depth_map_size=(85, 85)):
    """ Test the trained model with a new image. """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the trained model
    model = LandmarkToDepthCNN(landmark_dim=landmark_dim, landmark_coords=3, out_channels=1, depth_map_size=depth_map_size)
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model_state_dict = checkpoint['model_state_dict']  # Extract the model state dict
    
    # Load the state dict into the model
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    # Load and preprocess the image
    image = cv2.imread(image_path)
    landmarks_array = get_landmarks(image)
    landmarks_tensor = preprocess_landmarks(landmarks_array, landmark_dim).to(device)
    
    # Make the prediction
    with torch.no_grad():
        depth_map = model(landmarks_tensor)
    
    # Check the output
    depth_map_np = depth_map.squeeze().cpu().numpy()
    print(f"Depth map stats: min={depth_map_np.min()}, max={depth_map_np.max()}, mean={depth_map_np.mean()}")
    
    # Display results
    display_results(image, depth_map)

if __name__ == '__main__':
    image_path = 'train_data/images/image_299.jpg'
    model_path = 'checkpoint.pth'  # Ensure this is the correct path to your checkpoint file
    
    test(image_path, model_path)
