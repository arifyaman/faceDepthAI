import cv2
import os

def extract_frames(video_path, output_folder, frame_rate=30, width=720, height=1280):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return

    # Get the total number of frames in the video
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get the original frame rate of the video
    original_fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / frame_rate) if original_fps >= frame_rate else 1

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frame_count = 0
    saved_frame_count = 0

    while True:
        # Read a frame
        success, frame = video_capture.read()
        if not success:
            break  # End of video

        # Check if the current frame is at the specified interval
        if frame_count % frame_interval == 0:
            # Resize the frame
            frame_resized = cv2.resize(frame, (width, height))

            # Save frame as JPG
            output_path = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(output_path, frame_resized)
            saved_frame_count += 1

        frame_count += 1

        # Print progress every 100 frames (you can adjust the frequency)
        if frame_count % 100 == 0 or frame_count == total_frames:
            print(f"Processed {frame_count}/{total_frames} frames...")

    # Release the video capture object
    video_capture.release()
    print(f"Extracted {saved_frame_count} frames at {frame_rate} fps.")

# Usage
video_path = "calib.mp4"  # Path to the input video file
output_folder = "images"  # Folder to save the extracted frames
extract_frames(video_path, output_folder)
