import os
import io
import base64
from pymongo import MongoClient
import gridfs
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get MongoDB connection string from environment variable
mongodb_uri = os.getenv("MONGODB_URI")
if not mongodb_uri:
    print("Error: MongoDB URI not found. Please set the MONGODB_URI environment variable.")
    exit(1)

# Create a MongoDB client
client = MongoClient(mongodb_uri)

# Connect to the database
db = client["TIESImagesTest"]

# Create GridFS instance
fs = gridfs.GridFS(db)

def get_image_by_id(image_id):
    """
    Retrieve an image from GridFS by its image_id
    Returns the image as a bytes object or None if not found
    """
    # Find the file by image_id
    grid_out = fs.find_one({"image_id": image_id})
    
    if grid_out:
        # Return the image data
        return grid_out.read()
    return None

def get_image_base64(image_id):
    """
    Get image as base64 encoded string for embedding in HTML/markdown
    """
    image_data = get_image_by_id(image_id)
    if image_data:
        encoded = base64.b64encode(image_data).decode('utf-8')
        # Determine mime type based on filename in GridFS
        grid_out = fs.find_one({"image_id": image_id})
        if grid_out and grid_out.filename.lower().endswith('.gif'):
            mime = "image/gif"
        else:
            mime = "image/png"
        return f"data:{mime};base64,{encoded}"
    return None 