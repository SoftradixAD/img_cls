from flask import Flask, request, jsonify
import tensorflow as tf
from flask_cors import CORS  # Import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Load trained model
model = load_model("saved_model/cat_dog_unknown_classifier")

# Define class names
CLASS_NAMES = ['cat', 'dog', 'unknown']

def preprocess_image(img_path):
    """Ensure image preprocessing is identical in Flask and testing.py"""
    img = image.load_img(img_path, target_size=(256, 256), color_mode="grayscale")
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    print(f"Preprocessed Image Shape: {img_array.shape}")
    print(f"Min: {img_array.min()}, Max: {img_array.max()}")  # Debugging
    return img_array

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    file_path = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)
    file.save(file_path)

    print(f"Saved File Path: {file_path}")

    # Preprocess the image
    img_array = preprocess_image(file_path)

    # Get prediction
    prediction = model.predict(img_array)
    print(f"Raw Prediction: {prediction}")

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    os.remove(file_path)  # Clean up

    return jsonify({"prediction": predicted_class, "confidence": confidence})

if __name__ == '__main__':
    app.run(debug=True)
