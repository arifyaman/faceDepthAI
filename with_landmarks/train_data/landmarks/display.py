import numpy as np

def load_and_display_npy(file_path):
    try:
        # Load the .npy file
        data = np.load(file_path)

        # Display basic information about the data
        print(f"File: {file_path}")
        print(f"Data Shape: {data.shape}")
        print(f"Data Type: {data.dtype}")

        # Display the first 5 elements (or fewer if the array is smaller)
        print(data)

    except Exception as e:
        print(f"Error loading or displaying {file_path}: {e}")

# Example usage
landmarks_file = 'image_1089.npy'

load_and_display_npy(landmarks_file)