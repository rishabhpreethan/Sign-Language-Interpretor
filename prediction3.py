import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the trained model
model = load_model('hand_signs_model_big2.h5')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hand detector
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
drawing_spec_lines = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
drawing_spec_dots = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Define classes and standard size for model input
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'next']
standard_size = (48, 48)

# Variables to manage prediction timing and sentence building
update_interval = 1.5  # Interval to update predictions (in seconds)
last_update_time = time.time()
current_prediction = None
last_prediction = None
last_recognized = None  # To store the last recognized character before 'next'
sentence = ''  # This will store the accumulating sentence

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Create an image with a black background
    h, w, _ = img.shape
    img_landmarks = np.zeros((h, w, 3), dtype=np.uint8)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img_landmarks, hand_landmarks, mp_hands.HAND_CONNECTIONS, drawing_spec_dots, drawing_spec_lines)

            # Convert the landmark image to grayscale for processing
            black_image_gray = cv2.cvtColor(img_landmarks, cv2.COLOR_BGR2GRAY)
            hand_img = cv2.resize(black_image_gray, standard_size)

            # Normalize and prepare for model
            hand_img_normalized = np.expand_dims(hand_img, axis=-1) / 255.0
            hand_img_normalized = np.expand_dims(hand_img_normalized, axis=0)

            # Check if it's time to update prediction
            if time.time() - last_update_time > update_interval:
                # Make prediction
                prediction = model.predict(hand_img_normalized)
                class_id = np.argmax(prediction)
                confidence = np.max(prediction)
                current_prediction = classes[class_id]

                # Check for 'next' to update the sentence
                if current_prediction == 'next' and last_recognized is not None:
                    sentence += last_recognized
                    last_recognized = None
                elif current_prediction == 'space':
                    sentence += ' '
                else:
                    last_recognized = current_prediction

                last_update_time = time.time()

    # Display the current stable prediction and the accumulating sentence
    cv2.putText(img, f'Prediction: {current_prediction}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f'Sentence: {sentence}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the main image and the landmarks image
    cv2.imshow('Hand Sign Prediction', img)
    cv2.imshow('Hand Landmarks', img_landmarks)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
