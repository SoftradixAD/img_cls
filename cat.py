import os
import aiohttp
import asyncio

# Create a folder to store cat images
folder_name = "cat_images"
os.makedirs(folder_name, exist_ok=True)

API_URL = "https://api.thecatapi.com/v1/images/search"

async def fetch_image(session, i):
    """Fetch and save an image asynchronously."""
    async with session.get(API_URL) as response:
        if response.status == 200:
            data = await response.json()
            image_url = data[0]['url']
            
            if image_url.endswith((".jpg", ".jpeg", ".png")):
                print(f"Fetching image {i}: {image_url}")
                
                async with session.get(image_url) as img_response:
                    if img_response.status == 200:
                        img_data = await img_response.read()
                        img_extension = image_url.split('.')[-1]
                        img_path = os.path.join(folder_name, f"cat_image_{i}.{img_extension}")

                        with open(img_path, "wb") as img_file:
                            img_file.write(img_data)

                        print(f"Image {i} downloaded successfully and saved to {img_path}")
            else:
                print(f"Skipping non-static image: {image_url}")
        else:
            print(f"Failed to fetch image {i}")

async def download_images():
    """Download 1000 images asynchronously."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_image(session, i) for i in range(1, 1001)]
        await asyncio.gather(*tasks)

# Run the async function
asyncio.run(download_images())

