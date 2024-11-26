import os
from pycocotools.coco import COCO
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

class COCOCaptionDataset(Dataset):
    def __init__(self, images_dir, captions_path, transform=None):
        """
        Args:
            images_dir (str): Path to the directory containing images.
            captions_path (str): Path to the COCO captions file.
            transform (callable, optional): Transformation to apply to the images.
        """
        self.coco = COCO(captions_path)
        self.images_dir = images_dir
        self.transform = transform
        self.image_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Get image ID
        image_id = self.image_ids[idx]

        # Load image
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.images_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Load captions
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)
        captions = [ann['caption'] for ann in annotations]

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'captions': captions,
            'image_id': image_id
        }

def get_caption_dataloader(images_dir, captions_path, batch_size=4, shuffle=True, transform=None):
    """
    Creates a DataLoader for the COCO captions dataset.

    Args:
        images_dir (str): Path to the images directory.
        captions_path (str): Path to the COCO captions file.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        transform (callable, optional): Transformation to apply to the data.

    Returns:
        DataLoader: PyTorch DataLoader for the COCO captions dataset.
    """
    dataset = COCOCaptionDataset(images_dir, captions_path, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def collate_fn(batch):
    """
    Custom collate function to handle varying numbers of captions per image.
    """
    images = [item['image'] for item in batch]
    captions = [item['captions'] for item in batch]
    image_ids = [item['image_id'] for item in batch]

    # If using PyTorch tensors for images, stack them
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images)

    return {
        'images': images,
        'captions': captions,
        'image_ids': image_ids
    }
