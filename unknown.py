import os
import aiohttp
import asyncio

# Create a folder to store images
folder_name = "unknown"
os.makedirs(folder_name, exist_ok=True)

API_URL = "https://picsum.photos/256/256.jpg"  # Direct image URL

async def fetch_image(session, i):
    """Fetch and save an image asynchronously."""
    async with session.get(API_URL) as response:
        if response.status == 200:
            img_data = await response.read()  # Read image data
            img_path = os.path.join(folder_name, f"unknown_{i}.jpg")  # Save as .jpg

            with open(img_path, "wb") as img_file:
                img_file.write(img_data)

            print(f"Image {i} downloaded successfully and saved to {img_path}")
        else:
            print(f"Failed to fetch image {i}")

async def download_images():
    """Download 1000 images asynchronously."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_image(session, i) for i in range(1, 1001)]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    if os.name == "nt":  # Windows fix
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(download_images())
