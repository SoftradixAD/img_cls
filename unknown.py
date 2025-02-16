import requests
import os
import time

# Create the "unknown" folder if it doesn't exist
OUTPUT_FOLDER = "unknown"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Download 100 random images
for i in range(1, 1001):
    url = "https://picsum.photos/256"

    
    try:
        response = requests.get(url)  # Add timeout to prevent hanging

        if response.status_code == 200:
            file_path = os.path.join(OUTPUT_FOLDER, f"image_{i}.jpg")
            with open(file_path, "wb") as file:
                file.write(response.content)
            print(f"✅ Downloaded: {file_path}")
        else:
            print(f"❌ Failed to download image {i} (Status Code: {response.status_code})")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading image {i}: {e}")

    # time.sleep(2)  # Wait 2 seconds between requests
