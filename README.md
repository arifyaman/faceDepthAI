# Unreal Depth Frames

## Project Overview

**Unreal Depth Frames** is a Python-based project designed to process images, create a face mesh, generate a depth map, and return a depth frame for a given face. This tool is useful for the Metahuman performance in unreal engine

## Requirements

To run this project, you'll need to install the necessary dependencies. Follow the steps below to install the required Python packages.
Probably pip is already installed if you installed the python from offical source.

For detailed instructions on installing `pip`, refer to the official [pip installation guide](https://pip.pypa.io/en/stable/installation/).


## Installation

1. Clone the repository or download the source code.

2. Navigate to the project directory and install the required packages by running the following command: (use pip for python package manager)

   ```bash
   pip install -r requirements.txt
   ```

### Requirements File

The `requirements.txt` contains the following dependencies:

- opencv-python
- numpy
- mediapipe
- trimesh
- matplotlib
- openexr

These packages will be installed automatically when you run the above command.

## How It Works

1. The project reads an image file.
2. It detects the face within the image using the **MediaPipe** library and extracts the landmarks
3. A face mesh is generated using the facial landmarks from **MediaPipe**.
4. A depth map is created using image and the landmarks retrieved from the original image
5. **DepthMapProcessor** knows where your face is in the image and puts the created depth map excatly the same position from your image and creates the final depth map for the given dimensions.
5. The depth frame of the face is saved in .exr format (depth data in `Y` channel)

## Usage

Once you have the necessary dependencies installed

1. Go to `face_mesh` folder put your face image files (video frames) in to `face_mesh/images` directory.
2. Open `create_single_sample_and_display.py` file and set the `input_image_path` to one of your file name (ex: input_image_path = 'images/0022.jpg')
3. Run below to see created face depth map is overlapping with your face
```python
python create_single_sample_and_display.py
```
4. Play around the params to match the output face mask with original image
5. If the face mask matching with your face then copy the params and paste it into 
```python 
python convert_images_to_depth_maps.py
```
6. Run below script to get depth frames for all your files
```python
python convert_images_to_depth_maps.py
```
7. Corresponding depth maps will be saved in `face_mesh/depth_maps` folder. Then you can paste into depth frames into unreal engine.
