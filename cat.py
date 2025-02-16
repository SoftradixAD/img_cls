import requests
import os

# Create a folder to store cat images
folder_name = "cat_images"
os.makedirs(folder_name, exist_ok=True)

API_URL = "https://api.thecatapi.com/v1/images/search"

for i in range(1, 1001):  # Fetch 100 images
    response = requests.get(API_URL)
    
    if response.status_code == 200:
        data = response.json()
        image_url = data[0]['url']
        
        # Ensure it's a static image (JPEG or PNG)
        if image_url.endswith((".jpg", ".jpeg", ".png")):
            print(f"Fetching image {i}: {image_url}")
            
            # Download the image
            img_data = requests.get(image_url).content
            img_extension = image_url.split('.')[-1]  # Extract file extension
            img_path = os.path.join(folder_name, f"cat_image_{i}.{img_extension}")
            
            with open(img_path, "wb") as img_file:
                img_file.write(img_data)
            
            print(f"Image {i} downloaded successfully and saved to {img_path}")
        else:
            print(f"Skipping non-static image: {image_url}")
    else:
        print(f"Failed to fetch image {i}")
