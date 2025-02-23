import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("saved_model/cat_dog_unknown_classifier")
print(model.summary())
# Define class names (must match dataset order)
CLASS_NAMES = ['cat', 'dog', 'unknown']

def predict_image(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(256, 256), color_mode="grayscale")
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Get prediction
    prediction = model.predict(img_array)
    
    # Get predicted class
    predicted_class = CLASS_NAMES[np.argmax(prediction)]  # Fix here

    # Confidence score
    confidence = np.max(prediction)

    print(f"Prediction: {predicted_class} ({confidence:.2f})")

# Example usage
predict_image(r"C:\Users\AKSHAYDHADWAL\Documents\python_codes\unknown\unknown_997.jpg")
