from torchvision import transforms
from data_loader import get_caption_dataloader

 = '../coco_dataset/train2017'
captions_path = '../coco_dataset/annotations/captions_train2017.json'

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create DataLoader
dataloader = get_caption_dataloader(
    images_dir=images_dir,
    captions_path=captions_path,
    batch_size=1,
    transform=transform
)

# Iterate through DataLoader
for batch in dataloader:
    images = batch['images']
    captions = batch['captions']
    image_ids = batch['image_ids']

    print(f"Processed {len(images)} images with captions.")
    for img_caps in captions[:1]:  # Print captions for the first image in the batch
        print(f"Captions: {img_caps}")

    break




'''
# display the image with the caption using matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Display image
image = images[0].permute(1, 2, 0).numpy()
plt.imshow(image)
plt.axis('off')
plt.show()

# Display captions
for i, caption in enumerate(captions[0]):
    print(f"Caption {i+1}: {caption}")
'''