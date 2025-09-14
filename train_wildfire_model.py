import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# --- 1. Define Paths and Parameters ---
# This relative path will now work because the 'data' folder
# is in the same directory as this script.
DATA_DIR = 'data/train' 
CATEGORIES = ['nofire', 'fire']
IMG_SIZE = 128 # Resize images to 128x128

# --- 2. Load and Preprocess Images (IMPROVED VERSION) ---
def load_and_preprocess_data(data_dir):
    data = []
    for category in CATEGORIES:
        path = os.path.join(data_dir, category)
        class_num = CATEGORIES.index(category)
        print(f"Loading images from: {path}")
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                # Read the image
                img_array = cv2.imread(img_path)

                # --- THIS IS THE NEW, ROBUST CHECK ---
                if img_array is None:
                    print(f"  [Warning] Could not read image: {img_name}. Skipping file.")
                    continue # Skips to the next image in the loop

                # If the image is valid, resize and append it
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([resized_array, class_num])

            except Exception as e:
                # This will catch other errors, like if the file isn't an image at all
                print(f"  [Error] General error processing {img_name}: {e}. Skipping file.")
                continue
    return data
# THIS IS THE LINE THAT WAS MISSING. ADD IT BACK.
dataset = load_and_preprocess_data(DATA_DIR)

# --- 3. Prepare Data for the Model ---
X = []
y = []

# This loop will now work because 'dataset' has been created.
for features, label in dataset:
    X.append(features)
    y.append(label)

# Normalize pixel values to be between 0 and 1
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
y = np.array(y)

# Convert labels to one-hot encoding
y = to_categorical(y, num_classes=len(CATEGORIES))

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Data Shapes ---")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# --- 4. Build the CNN Model ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5), # Add dropout to prevent overfitting
    Dense(len(CATEGORIES), activation='softmax') # Softmax for multi-class classification
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 5. Train the Model ---
print("\n--- Training CNN Model ---")
history = model.fit(X_train, y_train,
                    epochs=15,
                    batch_size=32,
                    validation_data=(X_test, y_test))

# --- 6. Save the Trained Model ---
model.save('wildfire_cnn_model.h5')
print("\nWildfire detection model saved successfully!")