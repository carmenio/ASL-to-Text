import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

time.sleep(4)
print('\n\n')
# Create a directory to save frames
letter = input("What letter do you want to add training data for: ").lower()
name = input("What is your Name: ").lower()

output_dir = f"dataset/custom/{letter}"
os.makedirs(output_dir, exist_ok=True)

margin = 100
def extract_hand(frame):
    
    # Convert the frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect hands
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the bounding box of the detected hand
            h, w, c = frame.shape
            x_min = w
            y_min = h
            x_max = 0
            y_max = 0
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Calculate width and height
            width = x_max - x_min
            height = y_max - y_min

            # Determine the size for the square bounding box
            square_size = max(width, height) + 2 * margin  # Add margin to the square size

            # Center the square bounding box
            center_x = x_min + width // 2
            center_y = y_min + height // 2

            # Calculate new bounding box coordinates with margin
            x_min_square = max(0, center_x - square_size // 2)
            y_min_square = max(0, center_y - square_size // 2)
            x_max_square = min(w, center_x + square_size // 2)
            y_max_square = min(h, center_y + square_size // 2)

            # Draw the bounding box around the detected hand
            cv2.rectangle(frame, (x_min_square, y_min_square), (x_max_square, y_max_square), (0, 255, 0), 2)

            # Crop the image to just the hand with margin
            hand_img = frame[y_min_square:y_max_square, x_min_square:x_max_square]

            return hand_img, x_min_square, y_min_square, x_max_square, y_max_square
                
    return None, None, None, None, None

def record_frames():
    cap = cv2.VideoCapture(0)  # Capture video from webcam

    frame_count = 0
    recording = True

    while recording:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        hand_region, _, _, _, _ = extract_hand(frame)  # Process the frame for hand detection

        # Save the cropped hand region if detected
        if hand_region is not None:
            frame_filename = os.path.join(output_dir, f"hand_{frame_count:04d}_{name}.jpg")
            cv2.imwrite(frame_filename, hand_region)
            print(f"Saved: {frame_filename}")
            frame_count += 1
        
        # Display the frame
        cv2.imshow("Webcam Feed", frame)

        # Press 'q' to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            recording = False

    cap.release()
    cv2.destroyAllWindows()

# Start the recording when this function is called
record_frames()
