import os
from pycocotools.coco import COCO
import cv2
import torch
import numpy as np
import nltk
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from vocabulary import Vocabulary

nltk.download('punkt')


class COCOCaptionDataset(Dataset):

    def __init__(self, images_dir, captions_path, vocab_exists=False, transform=None):
        """
        Args:
            images_dir (str): Path to the directory containing images.
            captions_path (str): Path to the COCO captions file.
            vocab_exists: If False, create vocab from scratch and override any existing vocab_file.
                          If True, load vocab from existing vocab_file, if it exists.
            transform (callable, optional): Transformation to apply to the images.
        """

        self.coco = COCO(captions_path)
        self.ids = list(self.coco.anns.keys())

        self.images_dir = images_dir
        self.transform = transform
        self.image_ids = list(self.coco.imgs.keys())
        self.vocab_exists = vocab_exists

        # create vocabulary from the captions
        self.vocab = Vocabulary(annotations_file=captions_path, vocab_exists=vocab_exists)

        print("Obtaining caption lengths...")

        #  get list of tokens for each caption
        tokenized_captions = [
            nltk.tokenize.word_tokenize(
                str(self.coco.anns[self.ids[index]]["caption"]).lower()
            )
            for index in tqdm(np.arange(len(self.ids)))
        ]

        # get len of each caption
        self.caption_lengths = [len(token) for token in tokenized_captions]

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
        ann_id = self.ids[idx]
        captions = self.coco.anns[ann_id]["caption"]
        tokenized_caption = [self.vocab(self.vocab.start_word)]
        tokens = nltk.tokenize.word_tokenize(str(captions).lower())
        tokenized_caption.extend([self.vocab(token) for token in tokens])
        tokenized_caption.append(self.vocab(self.vocab.end_word))
        tokenized_caption = torch.Tensor(tokenized_caption).long()

        # Apply transformations
        image = self.transform(image)

        return {
            'image': image,
            'all_captions': captions,
            'tokenized_caption': tokenized_caption,
            'image_id': image_id
        }


def get_data_loader(images_dir, captions_path, vocab_exists=False, batch_size=4, shuffle=True, transform=None):
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
    dataset = COCOCaptionDataset(images_dir, captions_path, vocab_exists, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def collate_fn(batch):
    """
    Custom collate function to handle varying numbers of captions per image.
    """
    images = [item['image'] for item in batch]
    all_captions = [item['all_captions'] for item in batch]
    tokenized_caption = [item['tokenized_caption'] for item in batch]
    image_ids = [item['image_id'] for item in batch]

    # If using PyTorch tensors for images, stack them
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images)

    # Pad tokenized captions to the same length and stack into a tensor
    tokenized_caption = torch.nn.utils.rnn.pad_sequence(tokenized_caption, batch_first=True, padding_value=0)

    return {
        'images': images,
        'all_captions': all_captions,
        'tokenized_caption': tokenized_caption,
        'image_ids': image_ids
    }