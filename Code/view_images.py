from pycocotools.coco import COCO
import cv2
import matplotlib.pyplot as plt
import os

# Paths
images_dir = '../COCO_Data/train2017'  # Update with the correct image directory path
captions_path = '../COCO_Data/annotations/captions_train2017.json'

# Initialize COCO API
coco = COCO(captions_path)

# Function to extract image ID from filename
def get_image_id_from_filename(filename):
    return int(filename.split('.')[0])  # Extract numeric ID from '000000284286.jpg'

# Function to display images by filenames
def show_images_by_filenames(filenames, images_dir, coco):
    for filename in filenames:
        # Get image ID from filename
        image_id = get_image_id_from_filename(filename)
        
        # Load image info using COCO
        if image_id not in coco.imgs:
            print(f"Image ID {image_id} (from {filename}) does not exist in the COCO dataset.")
            continue
        image_info = coco.loadImgs(image_id)[0]
        image_path = os.path.join(images_dir, image_info['file_name'])
        
        # Check if the image file exists
        if not os.path.exists(image_path):
            print(f"Image file {filename} not found at {image_path}.")
            continue

        # Open image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display
        
        # Display image
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Image Filename: {filename}\nImage ID: {image_id}")
        plt.show()

# Example: Open images by filenames
example_filenames = ["000000284286.jpg", "000000325587.jpg", "000000158087.jpg"]
show_images_by_filenames(example_filenames, images_dir, coco)