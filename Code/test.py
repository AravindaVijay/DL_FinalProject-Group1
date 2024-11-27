from torchvision import transforms
from data_loader import get_data_loader

images_dir = '../coco_dataset/train2017'
captions_path = '../coco_dataset/annotations/captions_train2017.json'

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create DataLoader
dataloader = get_data_loader(
    images_dir=images_dir,
    captions_path=captions_path,
    vocab_exists=False,
    batch_size=1,
    transform=transform
)

# Iterate through DataLoader
for batch in dataloader:
    images = batch['images']
    all_captions = batch['all_captions']
    tokenized_caption = batch['tokenized_caption']
    image_ids = batch['image_ids']

    print(f"Processed {len(images)} images with captions.")
    for img_caps in all_captions[:1]:  # Print captions for the first image in the batch
        print(f"Captions: {img_caps}")
    print(f"Tokenized caption: {tokenized_caption}")

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