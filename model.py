import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import numpy as np
from tensorflow.keras.preprocessing import image

# Set dataset path
DATASET_PATH = "dataset"
IMG_SIZE = (256, 256)  # Image size
BATCH_SIZE = 16

# Load dataset with 3 classes (Cat, Dog, Unknown)
train_ds = keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="categorical"  # Convert labels to one-hot encoding
)

# Save class names before normalization
class_names = train_ds.class_names  
print(f"Train dataset loaded with classes: {class_names}")

# Load test dataset
test_ds = keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="categorical"
)

# Normalize pixel values (rescale 0-255 → 0-1)
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Define CNN model for 3-class classification
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(256, 256, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),  # Added dropout to prevent overfitting
    layers.Dense(3, activation="softmax")  # 3 output classes (Cat, Dog, Unknown)
])

# Compile model with Adam optimizer and categorical cross-entropy loss
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Reduced learning rate
              loss="categorical_crossentropy",  
              metrics=["accuracy"])

# Implement early stopping to avoid overfitting
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train model with validation set
EPOCHS = 20
model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=[early_stopping])

# Function to make predictions
def predict_image(img_path):
    img = image.load_img(img_path, color_mode="grayscale", target_size=IMG_SIZE)  # Load and resize
    img_array = image.img_to_array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dims for batch
    img_array /= 255.0  # Normalize (since we used Rescaling before)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]  # Use saved class names
    confidence = np.max(predictions)  # Confidence score

    print(f"🔍 Image: {os.path.basename(img_path)} → Prediction: {predicted_class} ({confidence:.2f})")

# Test the model on images from the test directory
test_images_dir = os.path.join(DATASET_PATH, "test")

if os.path.exists(test_images_dir):
    for category in os.listdir(test_images_dir):
        category_path = os.path.join(test_images_dir, category)
        if os.path.isdir(category_path):  # Ensure it's a folder
            print(f"\n📂 Testing images from category: {category}")
            for img_file in os.listdir(category_path): 
                img_path = os.path.join(category_path, img_file)
                predict_image(img_path)
else:
    print("❌ Test images directory not found!")



# Save the trained model
MODEL_PATH = "saved_model/cat_dog_unknown_classifier"
model.save(MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")

# Evaluate model on test dataset
test_loss, test_acc = model.evaluate(test_ds)
print(f"\n📊 Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

# Function to calculate model accuracy manually
def evaluate_model(dataset):
    correct = 0
    total = 0
    
    for images, labels in dataset:
        predictions = model.predict(images)  # Get predictions
        predicted_labels = np.argmax(predictions, axis=1)  # Convert softmax to class index
        true_labels = np.argmax(labels.numpy(), axis=1)  # Convert one-hot to class index
        
        correct += np.sum(predicted_labels == true_labels)  # Count correct predictions
        total += labels.shape[0]  # Total images
    
    accuracy = (correct / total) * 100
    print(f"\n📈 Model Correct Predictions: {correct}/{total} ({accuracy:.2f}%)")

# Call the function to print correct vs. incorrect
evaluate_model(test_ds)