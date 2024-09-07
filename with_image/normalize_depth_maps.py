import OpenEXR
import Imath
import numpy as np
import os

def write_exr_channel(file_path, channel_name, data_array, width, height):
    """Write a NumPy array to a specified channel in an EXR file."""
    header = OpenEXR.Header(width, height)
    header['channels'] = {channel_name: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    data_string = data_array.astype(np.float32).tobytes()
    
    # Open the EXR file
    exr_file = OpenEXR.OutputFile(file_path, header)
    try:
        exr_file.writePixels({channel_name: data_string})
    finally:
        exr_file.close()

def process_depth_map(input_file_path, output_file_path, depth_channel='Y'):
    try:
        print(f"Opening EXR file: {input_file_path}")
        exr_file = OpenEXR.InputFile(input_file_path)

        # Get the header information
        header = exr_file.header()

        # Display channel information
        channels = header['channels'].keys()
        print("Channels available in the EXR file:", channels)

        # Define the pixel type (FLOAT)
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)

        # Read a specific channel
        if depth_channel in channels:
            print(f"Processing channel: {depth_channel}")
            depth_channel_data = exr_file.channel(depth_channel, pixel_type)
            
            # Convert the channel data to a numpy array
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1
            depth_data = np.frombuffer(depth_channel_data, dtype=np.float32).reshape((height, width))
            
            # Ensure depth_data is writable
            depth_data = np.copy(depth_data)

            # Process the depth data: set values to 100 if they are 0 or greater than 80
            print("Processing depth data...")
            depth_data[(depth_data == 0) | (depth_data > 80)] = 100

            # Display statistics about the processed depth data
            print(f"Processed Depth data min: {depth_data.min()}, max: {depth_data.max()}")

            # Save the modified depth data to a new EXR file
            print(f"Preparing to save processed EXR file: {output_file_path}")
            write_exr_channel(output_file_path, depth_channel, depth_data, width, height)
            
            print(f"Processed depth map saved to {output_file_path}")

        else:
            print(f"Depth channel ('{depth_channel}') not found in the EXR file.")

    except Exception as e:
        print(f"An error occurred while processing {input_file_path}: {e}")

def process_all_exr_files(input_folder, output_folder, depth_channel='Y'):
    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate over all EXR files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith('.exr'):
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)
            print(f"Processing {input_file_path}...")
            process_depth_map(input_file_path, output_file_path, depth_channel)

if __name__ == "__main__":
    input_folder = 'depth_maps'
    output_folder = 'normalized_depth_maps'

    process_all_exr_files(input_folder, output_folder)
