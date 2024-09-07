import os

# Directory containing the .exr files
directory = 'temp_dir'
ext = '.exr'
start = 2238

# Get a list of all .exr files in the directory
files = [f for f in os.listdir(directory) if f.endswith(ext)]

# Loop through the files and rename them
for filename in files:
    # Extract the number from the original filename
    num_str = filename.split('_')[1].split('.')[0]
    number = int(num_str) + start  # Convert to integer and adjust to start from 1

    # New filename format
    new_name = f'image_{number}{ext}'
    
    # Full old and new file paths
    old_file = os.path.join(directory, filename)
    new_file = os.path.join(directory, new_name)
    
    # Rename the file
    os.rename(old_file, new_file)

print("Renaming complete!")