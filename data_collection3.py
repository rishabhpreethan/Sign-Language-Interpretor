import cv2
import os
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hand Solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
drawing_spec_lines = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=13, circle_radius=4)
drawing_spec_dots = mp_drawing.DrawingSpec(color=(55, 255, 255), thickness=7, circle_radius=4)
hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def collect_images_for_letter(letter, num_images=300, standard_size=(48, 48)):
    cap = cv2.VideoCapture(0)

    # Create a directory for the letter if it doesn't exist
    if not os.path.exists(letter):
        os.makedirs(letter)

    count = 0
    print(f"Collecting images for letter: {letter}. Press 'c' to capture.")

    while cap.isOpened() and count < num_images:
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)  # Flip the image for a later selfie-view display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = image.shape
                black_image = np.zeros((h, w, 3), dtype=np.uint8)
                mp_drawing.draw_landmarks(black_image, hand_landmarks, mp_hands.HAND_CONNECTIONS, drawing_spec_dots, drawing_spec_lines)

                # Use the entire image for consistent training data
                resized_image = cv2.resize(black_image, standard_size)
                cv2.imshow('Collecting Images', resized_image)

                if cv2.waitKey(5) & 0xFF == ord('c'):
                    img_name = f"{letter}/{letter}_{count}.jpg"
                    cv2.imwrite(img_name, resized_image)
                    print(f"{img_name} written!")
                    count += 1
        else:
            print("No hand detected.")

    cap.release()
    cv2.destroyAllWindows()

# Include all letters G to Z and special commands
# letters = ['G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'next']
letters = ['I love you', 'How are you', 'Good', 'Bad']
for letter in letters:
    input(f"Press Enter to start collecting images for letter {letter}...")
    collect_images_for_letter(letter, num_images=1000)
