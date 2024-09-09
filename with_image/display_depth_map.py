import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt

# Replace with the path to your EXR file

file_path = "depth_000000.exr"
#file_path = "calculate/output_depth.exr"
depth_channel = "Y"


# Open the EXR file
exr_file = OpenEXR.InputFile(file_path)

# Get the header information
header = exr_file.header()

# Display channel information
channels = header['channels'].keys()
print("Channels available in the EXR file:", channels)

# Define the pixel type (FLOAT)
pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)

# Read a specific channel (e.g., 'Z' for depth)
if depth_channel in channels:
    depth_channel = exr_file.channel(depth_channel, pixel_type)
    # Convert the channel data to a numpy array
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    depth_data = np.frombuffer(depth_channel, dtype=np.float32).reshape((height, width))

    # Display statistics about the depth data
    print(f"Depth data min: {depth_data.min()}, max: {depth_data.max()}")

    # Visualize the depth data using a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(depth_data, cmap='viridis')
    plt.colorbar(label='Depth Value')
    plt.title('Depth Map Visualization')
    plt.show()

else:
    print("Depth channel ('Z') not found in the EXR file.")