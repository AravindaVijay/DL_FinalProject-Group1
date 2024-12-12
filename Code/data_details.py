import json

captions_path = "../COCO_Data/annotations/captions_train2017.json"
with open(captions_path, 'r') as f:
    data = json.load(f)

images = data['images']
annotations = data['annotations']

num_images = len(images)

captions_per_image = {}
for annotation in annotations:
    image_id = annotation['image_id']
    caption = annotation['caption']
    if image_id not in captions_per_image:
        captions_per_image[image_id] = []
    captions_per_image[image_id].append(caption)

num_captions = {image_id: len(captions) for image_id, captions in captions_per_image.items()}

unique_captions_count = set(num_captions.values())

classes = data.get('categories', None)

print(f"Total number of images: {num_images}")
print(f"Unique counts of captions per image: {unique_captions_count}")
if classes:
    print(f"Number of classes: {len(classes)}")
    print(f"Classes: {[category['name'] for category in classes]}")
else:
    print("No classes or sections found in the dataset.")