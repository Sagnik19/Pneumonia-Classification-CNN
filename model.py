import os
import numpy as np
from tensorflow.keras.models import load_model

# Get current folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model path
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "trained",
    "pneumonia_cnn_final.keras"
)

# Load model
model = load_model(MODEL_PATH)


def predict_image(img_array):
    """
    Predict Pneumonia or Normal
    """

    prediction = model.predict(img_array, verbose=0)

    confidence = float(prediction[0][0])

    if confidence > 0.5:
        label = "PNEUMONIA"
    else:
        label = "NORMAL"

    return label, confidence