# Unreal Depth Frames

[Youtube demo](https://www.youtube.com/watch?v=fsi7dAxHAuk)

## Project Overview

**Unreal Depth Frames** is a Python-based project designed to process images, create a face mesh, generate a depth map, and return a depth frame for a given face. This tool is useful for the Metahuman performance in Unreal Engine.

## Requirements

To run this project, you'll need to install the necessary dependencies. Follow the steps below to install the required Python packages. `pip` is likely already installed if you installed Python from the official source.

For detailed instructions on installing `pip`, refer to the official [pip installation guide](https://pip.pypa.io/en/stable/installation/).

## Installation

1. Clone the repository or download the source code.
2. Navigate to the project directory and install the required packages by running the following command: 

   ```python
   pip install -r requirements.txt
   ```

### Requirements File

The `requirements.txt` contains the following dependencies & versions: You can create a virtual env to avoid dependency conflicts (Recommended)

- opencv-python==4.10.0.84
- numpy==1.26.4
- mediapipe==0.10.18
- trimesh==4.5.2
- matplotlib==3.9.2
- openexr==3.3.1


These packages will be installed automatically when you run the above command.

## How It Works

1. The project reads an image file.
2. It detects the face within the image using the **MediaPipe** library and extracts the landmarks.
3. A face mesh is generated using the facial landmarks from **MediaPipe**.
4. A depth map is created using the image and the landmarks retrieved from the original image.
5. **DepthMapProcessor** knows where your face is in the image and places the created depth map exactly in the same position, creating the final depth map for the given dimensions.
6. The depth frame of the face is saved in `.exr` format (depth data in the `Y` channel).

## Usage

Once you have the necessary dependencies installed:

1. **Extract Video Frames**  

   **Recommended Video Settings**:  
      - **Aspect Ratio**: 9:16  
      - **Frame Rate**: 30 fps or higher 
      
   To start, extract frames from a video file using the `extract_frames_from_video.py` script. Set the `video_path` to the path of your video file (e.g., `video_path = "calib.mp4"`) and run the script. This will extract the frames into the `face_mesh/images` directory.

2. Go to the `face_mesh` folder and place your extracted face image files (video frames) into the `face_mesh/images` directory.

3. Open the `create_single_sample_and_display.py` file and set the `input_image_path` to one of your file names (e.g., `input_image_path = 'images/0022.jpg'`).

4. Run the script to see the created face depth map overlapping with your face.

```python
   python create_single_sample_and_display.py
   ```

5. Play around with the parameters to match the output face mask with the original image.

6. If the face mask matches your face, copy the parameters and paste them into `convert_images_to_depth_maps.py`.

7. Run the script to get depth frames for all your files.

```python
   python convert_images_to_depth_maps.py
   ```

8. The corresponding depth maps will be saved in the `face_mesh/depth_maps` folder. You can then import the depth frames into Unreal Engine.
