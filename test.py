import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow as tf

# Load the image
image_path = "1.png"  # Replace with your actual image path
image = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image, channels=3)  # Use decode_png() for PNG

# Resize to 256x256
image_resized = tf.image.resize(image, [256, 256])

# Convert to a tensor (optional, already a tensor)
image_resized = tf.convert_to_tensor(image_resized)

# Print tensor details
print("Resized Tensor shape:", image_resized.shape)
print("Tensor dtype:", image_resized.dtype)



# Convert to grayscale
image_gray = tf.image.rgb_to_grayscale(image)  # Shape becomes (H, W, 1)

# Resize to 256x256
image_gray_resized = tf.image.resize(image_gray, [256, 256])

# Print tensor details
print("Resized Grayscale Tensor shape:", image_gray_resized.shape)
print("Tensor dtype:", image_gray_resized.dtype)


# Convert to uint8 format for saving
image_gray_uint8 = tf.image.convert_image_dtype(image_gray_resized, dtype=tf.uint8)

# Encode as PNG
encoded_image = tf.io.encode_png(image_gray_uint8)

# Save to file as "2.png"
tf.io.write_file("2.png", encoded_image)

print("Grayscale image saved as 2.png")