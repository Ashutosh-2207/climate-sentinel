# backend/model_handler.py

from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import io

# --- Configuration ---
IMG_SIZE = 128
MODEL_PATH = 'wildfire_cnn_model.h5' # Assumes model is in the same 'backend' folder
WILDFIRE_MODEL = None

# --- Load Model on Startup ---
def load_wildfire_model():
    """Loads the wildfire detection model into memory."""
    global WILDFIRE_MODEL
    print("Loading Wildfire CNN model into memory...")
    try:
        WILDFIRE_MODEL = load_model(MODEL_PATH)
        print("Wildfire CNN model loaded successfully!")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load wildfire model at '{MODEL_PATH}'.")
        print(f"Error: {e}")
        WILDFIRE_MODEL = None

# --- Prediction Function ---
def predict_wildfire_from_image(image_bytes: bytes):
    """
    Predicts wildfire presence from image bytes using the loaded CNN model.
    Returns a dictionary with the prediction and confidence score.
    """
    if WILDFIRE_MODEL is None:
        return {"error": "Model is not loaded."}

    try:
        # Convert bytes to a NumPy array
        image = Image.open(io.BytesIO(image_bytes))
        # Convert PIL image to OpenCV format (RGB -> BGR)
        image_np = np.array(image)
        
        # Ensure image is 3 channels (RGB)
        if len(image_np.shape) == 2: # Grayscale image
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        elif image_np.shape[2] == 4: # RGBA image
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
            
        # Preprocess the image for the model
        resized_array = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE))
        scaled_array = resized_array / 255.0
        input_data = np.expand_dims(scaled_array, axis=0) # Add batch dimension

        # Make prediction
        prediction = WILDFIRE_MODEL.predict(input_data)
        
        # Interpret the prediction
        # prediction[0] will be like [prob_nofire, prob_fire]
        confidence_fire = float(prediction[0][1])
        
        if confidence_fire > 0.5:
            label = "Fire Detected"
        else:
            label = "No Fire Detected"
        
        return {
            "prediction": label,
            "confidence": f"{confidence_fire:.2%}"
        }

    except Exception as e:
        return {"error": f"Error processing image: {e}"}