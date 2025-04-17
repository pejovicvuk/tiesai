import os
import requests
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import concurrent.futures

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# Define the directory where your images are stored - updated with your path
image_dir = r"C:\Users\pejov\source\local\zendesk documentation\images"

# Function to upload an image to OpenAI
def upload_image_to_openai(image_path):
    """Upload an image to OpenAI and return the file ID"""
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    # Determine the MIME type based on file extension
    file_extension = os.path.splitext(image_path)[1].lower()
    mime_type = "image/png"  # Default
    if file_extension == ".jpg" or file_extension == ".jpeg":
        mime_type = "image/jpeg"
    elif file_extension == ".gif":
        mime_type = "image/gif"
    
    try:
        with open(image_path, "rb") as image_file:
            files = {
                "file": (os.path.basename(image_path), image_file, mime_type),
                "purpose": (None, "assistants")
            }
            
            response = requests.post(
                "https://api.openai.com/v1/files",
                headers=headers,
                files=files
            )
            
            if response.status_code == 200:
                return response.json()["id"]
            else:
                print(f"Error uploading {image_path}: {response.text}")
                return None
    except Exception as e:
        print(f"Exception uploading {image_path}: {str(e)}")
        return None

# Find all image files recursively (including in subfolders)
print(f"Scanning for images in: {image_dir}")
image_paths = []
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            full_path = os.path.join(root, file)
            image_paths.append(full_path)
            # Print the folder structure to verify
            relative_path = os.path.relpath(full_path, image_dir)
            print(f"Found image: {relative_path}")

print(f"Found {len(image_paths)} images to upload")

# Check if we have an existing mapping file to resume from
image_mapping = {}
if os.path.exists("image_mapping.json"):
    try:
        with open("image_mapping.json", "r") as f:
            image_mapping = json.load(f)
        print(f"Loaded existing mapping with {len(image_mapping)} entries")
    except:
        print("Could not load existing mapping, starting fresh")

# Filter out images that have already been uploaded
images_to_upload = []
for path in image_paths:
    # Extract the original image ID from the filename
    original_id = os.path.splitext(os.path.basename(path))[0]
    if original_id not in image_mapping:
        images_to_upload.append(path)

print(f"{len(images_to_upload)} images need to be uploaded")

# Function to process a batch of images
def process_batch(batch):
    results = {}
    for path in batch:
        original_id = os.path.splitext(os.path.basename(path))[0]
        print(f"Uploading: {original_id} from {path}")
        openai_file_id = upload_image_to_openai(path)
        if openai_file_id:
            results[original_id] = openai_file_id
            # Save progress after each successful upload
            with open("image_mapping_temp.json", "w") as f:
                temp_mapping = {**image_mapping, **results}
                json.dump(temp_mapping, f, indent=2)
        # Sleep to respect rate limits (100 files per minute = ~0.6s per file)
        time.sleep(0.7)
    return results

# Split images into batches for parallel processing
# Using smaller batches to avoid overwhelming the API
batch_size = 10
batches = [images_to_upload[i:i + batch_size] for i in range(0, len(images_to_upload), batch_size)]

# Process batches with a thread pool
# Using max_workers=3 to stay well within rate limits
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(process_batch, batch) for batch in batches]
    
    # Collect results as they complete
    for future in concurrent.futures.as_completed(futures):
        batch_results = future.result()
        # Update the main mapping
        image_mapping.update(batch_results)
        
        # Save progress
        with open("image_mapping.json", "w") as f:
            json.dump(image_mapping, f, indent=2)
        
        print(f"Progress: {len(image_mapping)}/{len(image_paths)} images processed")

print(f"Upload complete! {len(image_mapping)}/{len(image_paths)} images processed")
print(f"Mapping saved to image_mapping.json")