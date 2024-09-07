import os
from rembg import remove
from PIL import Image

def remove_background(input_folder, output_folder):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".jpg")
            
            # Load the image
            with Image.open(input_path) as img:
                # Remove the background
                output = remove(img)

                # Create a new black background image
                black_bg = Image.new("RGB", output.size, (0, 0, 0))
                
                # Composite the original image onto the black background
                output = Image.composite(output, black_bg, output.split()[3])  # output.split()[3] is the alpha channel
                
                # Save the result as a JPEG
                output.save(output_path, "JPEG")
            
            print(f"Processed: {filename}")

# Usage
input_folder = 'images'
output_folder = 'without_background_images'

remove_background(input_folder, output_folder)
