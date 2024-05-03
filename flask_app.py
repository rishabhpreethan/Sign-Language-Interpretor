from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
import mediapipe as mp
from textblob import Word
import requests

app = Flask(__name__)

# Load the trained model
model = load_model('hand_signs_model_final1.h5')

# Initialize MediaPipe hand detector
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
drawing_spec_lines = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
drawing_spec_dots = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Define classes and standard size for model input
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'next', 'HOW ARE YOU', 'GOOD', 'BAD', 'I LOVE YOU']
standard_size = (48, 48)

# Variables to manage prediction timing and sentence building
update_interval = 3  # Interval to update predictions (in seconds)
last_update_time = time.time()
current_prediction = None
last_prediction = None
last_recognized = None  # To store the last recognized character before 'next'
sentence = ''  # This will store the accumulating sentence

# Initialize webcam
cap = cv2.VideoCapture(0)

def get_suggestions(prefix):
    if prefix:
        url = f"https://api.datamuse.com/sug?s={prefix}"
        response = requests.get(url)
        if response.status_code == 200:
            suggestions = [item['word'] for item in response.json()]
            return suggestions[:3]  # Return top 3 suggestions
    return []

def generate_frames():
    global cap, last_update_time, current_prediction, last_prediction, last_recognized, sentence
    while True:
        success, img = cap.read()
        if not success:
            continue

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        h, w, _ = img.shape
        img_landmarks = np.zeros((h, w, 3), dtype=np.uint8)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img_landmarks, hand_landmarks, mp_hands.HAND_CONNECTIONS, drawing_spec_dots, drawing_spec_lines)

                # Prediction and sentence update logic
                black_image_gray = cv2.cvtColor(img_landmarks, cv2.COLOR_BGR2GRAY)
                hand_img = cv2.resize(black_image_gray, standard_size)
                hand_img_normalized = np.expand_dims(hand_img, axis=-1) / 255.0
                hand_img_normalized = np.expand_dims(hand_img_normalized, axis=0)

                if time.time() - last_update_time > update_interval:
                    prediction = model.predict(hand_img_normalized)
                    class_id = np.argmax(prediction)
                    confidence = np.max(prediction)
                    current_prediction = classes[class_id]

                    if current_prediction == 'next' and last_recognized is not None:
                        sentence += last_recognized
                        last_recognized = None
                    elif current_prediction == 'space':
                        sentence += ' '
                    else:
                        last_recognized = current_prediction

                    last_update_time = time.time()

        # Encode both camera image and landmarks image
        combined_img = np.hstack((img, img_landmarks))

        # Encode combined image
        ret, frame = cv2.imencode('.jpg', combined_img)
        frame = frame.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html', prediction=current_prediction, sentence=sentence)

@app.route('/get_suggestions')
def get_suggestions_route():
    prefix = request.args.get('last_char', '').strip()
    suggestions = get_suggestions(prefix)
    return jsonify(suggestions)

@app.route('/add_to_sentence', methods=['GET'])
def add_to_sentence():
    suggestion = request.args.get('suggestion', '')
    global sentence
    if sentence and suggestion:
        words = sentence.strip().split()
        if words:
            words[-1] = suggestion  # Replace the last word with the full suggestion
            sentence = ' '.join(words)
        else:
            sentence = suggestion
    elif not sentence:
        sentence = suggestion  # For the first word
    return jsonify(success=True)

@app.route('/get_data')
def get_data():
    return jsonify(prediction=current_prediction, sentence=sentence)

@app.route('/delete', methods=['POST'])
def delete():
    global sentence
    if request.method == 'POST':
        if len(sentence) > 0:
            sentence = sentence[:-1]  # Delete the last character
    return render_template('index.html', prediction=current_prediction, sentence=sentence)

@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    global sentence
    sentence = ''  # Reset the sentence to an empty string
    return jsonify(success=True)  # Respond with success


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, threaded=True)

# Cleanup
cap.release()
cv2.destroyAllWindows()