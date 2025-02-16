import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
import os

# Load the trained model
model = tf.keras.models.load_model("cat_dog_classifier.h5")

# Initialize Flask app
app = Flask(__name__)

# Create an upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_image(img_path):
    """Predict whether an image contains a cat or a dog."""
    img = image.load_img(img_path, target_size=(256, 256), color_mode="grayscale")
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)[0][0]
    label = "Dog" if prediction > 0.5 else "Cat"
    
    return {"label": label, "confidence": float(prediction)}

@app.route("/predict", methods=["POST"])
def predict():
    """Handle POST request for image classification."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save the file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Predict
    result = predict_image(file_path)

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
