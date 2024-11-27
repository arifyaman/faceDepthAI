import cv2
import os

class FrameExtractor:
    def __init__(self, video_path, output_folder, frame_rate=30, width=1280, height=720):
        self.video_path = video_path
        self.output_folder = output_folder
        self.frame_rate = frame_rate
        self.width = width
        self.height = height

    def extract_frames(self):
        # Open the video file
        video_capture = cv2.VideoCapture(self.video_path)
        
        # Check if video opened successfully
        if not video_capture.isOpened():
            print(f"Error: Could not open video {self.video_path}.")
            return

        # Get the original frame rate of the video
        original_fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_interval = int(original_fps / self.frame_rate) if original_fps >= self.frame_rate else 1

        # Create the output directory if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
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
                frame_resized = cv2.resize(frame, (self.width, self.height))

                # Save frame as JPG
                output_path = os.path.join(self.output_folder, f"frame_{saved_frame_count:04d}.jpg")
                cv2.imwrite(output_path, frame_resized)
                saved_frame_count += 1

            frame_count += 1

        # Release the video capture object
        video_capture.release()
        print(f"Extracted {saved_frame_count} frames at {self.frame_rate} fps in '{self.output_folder}'.")
