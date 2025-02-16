import tensorflow as tf
import os

# Define input and output directories
INPUT_FOLDER = "unknown"
OUTPUT_FOLDER = "grayscale_unknown"

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def convert_images_to_grayscale():
    """Reads images from INPUT_FOLDER, resizes to 256x256 if needed, converts to grayscale, and saves to OUTPUT_FOLDER."""
    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):  # Process only image files
            image_path = os.path.join(INPUT_FOLDER, filename)
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3)  # Ensure RGB

            # Convert to grayscale
            image_gray = tf.image.rgb_to_grayscale(image)

            # Convert to uint8 for saving
            image_gray_uint8 = tf.image.convert_image_dtype(image_gray, dtype=tf.uint8)

            # Encode and save
            output_path = os.path.join(OUTPUT_FOLDER, filename)  # Save with original filename
            encoded_image = tf.io.encode_png(image_gray_uint8)
            tf.io.write_file(output_path, encoded_image)

            print(f"Processed: {filename} -> {output_path}")

if __name__ == "__main__":
    convert_images_to_grayscale()
    print("✅ All images resized to 256x256 and converted to grayscale successfully!")
