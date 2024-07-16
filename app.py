import cv2
import mediapipe as mp
from PIL import Image
import numpy as np

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize OpenCV capture
cap = cv2.VideoCapture(0)

# Load the earring image
earring_image = Image.open('data/earring1.png')  # Path to your earring image
earring_image = earring_image.convert("RGBA")

# Function to overlay earring on the frame
def overlay_image(background, overlay, x, y):
    background.paste(overlay, (x, y), overlay)
    return background

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Convert the frame to PIL Image for easier manipulation
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Check if any face is detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get 3D coordinates for the left and right ear landmarks
            left_ear_3d = face_landmarks.landmark[177]
            right_ear_3d = face_landmarks.landmark[401]

            # Convert 3D coordinates to 2D pixel coordinates
            left_ear_x = int(left_ear_3d.x * frame.shape[1])
            left_ear_y = int(left_ear_3d.y * frame.shape[0])
            left_ear_z = left_ear_3d.z
            right_ear_x = int(right_ear_3d.x * frame.shape[1])
            right_ear_y = int(right_ear_3d.y * frame.shape[0])
            right_ear_z = right_ear_3d.z

            # Adjust the size of the earring image
            earring_size = 12  # Define the size of the earring
            resized_earring = earring_image.resize((earring_size, earring_size), Image.Resampling.LANCZOS)

            # Adjustments to place the earrings behind the cheeks and near the ears
            left_ear_x_adjusted = left_ear_x - resized_earring.width // 2 
            left_ear_y_adjusted = left_ear_y + resized_earring.height // 2
            right_ear_x_adjusted = right_ear_x - resized_earring.width // 2 
            right_ear_y_adjusted = right_ear_y + resized_earring.height // 2
            #left_ear_z_adjusted = left_ear_z + resized_earring.height // 2
            #right_ear_z_adjusted = left_ear_z + resized_earring.height // 2


            # Overlay the earring on the left earlobe
            pil_image = overlay_image(pil_image, resized_earring, left_ear_x_adjusted, left_ear_y_adjusted)

            # Overlay the earring on the right earlobe
            pil_image = overlay_image(pil_image, resized_earring, right_ear_x_adjusted, right_ear_y_adjusted)

    # Convert the PIL Image back to OpenCV format
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    cv2.imshow('Virtual Try-On', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()