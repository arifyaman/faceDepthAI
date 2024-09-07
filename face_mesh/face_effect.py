import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe FaceMesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Function to apply color to specific face regions
def apply_color_to_region(image, region_points, color):
    region_points = np.array(region_points, np.int32)
    cv2.fillPoly(image, [region_points], color)

# Drawing specification for mediapipe (can customize appearance)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Load the image (change this to your image path)
image_path = 'image_497.jpg'  # Change this to your image path
image = cv2.imread(image_path)
h, w, _ = image.shape

# Initialize FaceMesh for static image mode
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:

    # Process the image to detect face landmarks
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Check if any face landmarks were detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get normalized landmark coordinates
            landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]

            # Example: Left Eye (landmarks around eye - indices 33 to 133)
            left_eye_indices = list(range(33, 133))
            left_eye_pts = [landmarks[i] for i in left_eye_indices]
            apply_color_to_region(image, left_eye_pts, (0, 255, 0))  # Green for left eye

            # Example: Right Eye (landmarks around eye - indices 362 to 382)
            right_eye_indices = list(range(362, 382))
            right_eye_pts = [landmarks[i] for i in right_eye_indices]
            apply_color_to_region(image, right_eye_pts, (0, 0, 255))  # Red for right eye

            # Example: Left Cheek (landmarks roughly from 115 to 130)
            left_cheek_indices = [115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129]
            left_cheek_pts = [landmarks[i] for i in left_cheek_indices]
            apply_color_to_region(image, left_cheek_pts, (255, 0, 0))  # Blue for left cheek

            # Example: Right Cheek (landmarks roughly from 345 to 356)
            right_cheek_indices = [345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355]
            right_cheek_pts = [landmarks[i] for i in right_cheek_indices]
            apply_color_to_region(image, right_cheek_pts, (0, 255, 255))  # Yellow for right cheek

            # Draw face mesh tesselation and contours
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

# Save and display the output image with effects
cv2.imshow('Face with Effects', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
