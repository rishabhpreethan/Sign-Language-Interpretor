import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from PIL import Image

# Path to the dataset
dataset_path = "C:/Users/Ritesh/Documents/github/new_sign_interpretor/dataset2"
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'next', 'How are you', 'Good', 'Bad', 'I love you']  # Ensure you include all classes

# Initialize lists to hold the image data and labels
image_files = []
labels = []

# Image parameters
img_width, img_height = 48, 48  # Adjusted to match prediction input

# Load images and labels from the filesystem
for label, class_name in enumerate(classes):
    class_path = os.path.join(dataset_path, class_name)
    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        img = Image.open(img_path).resize((img_width, img_height)).convert('L')
        img_array = np.array(img) / 255.0
        image_files.append(img_array)
        labels.append(label)

# Convert lists to numpy arrays and prepare for training
X = np.array(image_files).reshape(-1, img_width, img_height, 1)
y = to_categorical(np.array(labels), num_classes=len(classes))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model architecture
model = Sequential([
    Conv2D(128, (3, 3), activation='relu', input_shape=(img_width, img_height, 1)),
    MaxPooling2D((2, 2)), Dropout(0.4),
    Conv2D(256, (3, 3), activation='relu'), MaxPooling2D((2, 2)), Dropout(0.4),
    Conv2D(512, (3, 3), activation='relu'), MaxPooling2D((2, 2)), Dropout(0.4),
    Conv2D(512, (3, 3), activation='relu'), MaxPooling2D((2, 2)), Dropout(0.4),
    Flatten(),
    Dense(512, activation='relu'), Dropout(0.4),
    Dense(64, activation='relu'), Dropout(0.2),
    Dense(256, activation='relu'), Dropout(0.3),
    Dense(64, activation='relu'), Dropout(0.2),
    Dense(256, activation='relu'), Dropout(0.3),
    Dense(32, activation='softmax')
])

# Compile and train the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=128, epochs=12)
model.save('hand_signs_model_final1.h5')

print(model.summary())