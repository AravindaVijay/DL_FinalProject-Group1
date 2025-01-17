##########################################################################
###################### Vision Transformer and Custom Decoder #############
##########################################################################

import os
import torch
import random
import nltk
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import timm
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
from data_loader import get_data_loader
from vocabulary import Vocabulary


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class CaptionVocabulary:
    def __init__(self, annotation_file, threshold=5):
        self.token_to_id = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.word_counts = {}
        self.threshold = threshold
        self._construct_vocab(annotation_file)

    def _construct_vocab(self, annotation_file):
        coco = COCO(annotation_file)
        for ann_id in coco.anns:
            caption = coco.anns[ann_id]["caption"]
            for word in nltk.word_tokenize(caption.lower()):
                self.word_counts[word] = self.word_counts.get(word, 0) + 1
        for word, count in self.word_counts.items():
            if count >= self.threshold:
                idx = len(self.token_to_id)
                self.token_to_id[word] = idx
                self.id_to_token[idx] = word

    def __call__(self, word):
        return self.token_to_id.get(word, self.token_to_id["<unk>"])

    def __len__(self):
        return len(self.token_to_id)

class CaptionDataset(Dataset):
    def __init__(self, image_dir, annotation_path, vocab, transform=None):
        self.coco = COCO(annotation_path)
        self.image_ids = list(self.coco.imgs.keys())
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        caption_ids = self.coco.getAnnIds(imgIds=image_id)
        captions = [self.coco.anns[ann_id]["caption"] for ann_id in caption_ids]
        tokens = [self.vocab("<start>")] + \
                 [self.vocab(word) for word in nltk.word_tokenize(captions[0].lower())] + \
                 [self.vocab("<end>")]

        return {"image": image, "caption": torch.tensor(tokens, dtype=torch.long)}

def create_data_loader(img_dir, ann_file, vocab, batch_size, image_transform):
    dataset = CaptionDataset(img_dir, ann_file, vocab, image_transform)
    
    def custom_collate_fn(batch):
        images = torch.stack([item["image"] for item in batch])
        captions = [item["caption"] for item in batch]
        captions_padded = pad_sequence(captions, batch_first=True, padding_value=0)
        return {"image": images, "caption": captions_padded}
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)


class TransformerEncoder(nn.Module):
    def __init__(self, output_dim):
        super(TransformerEncoder, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.model.head = nn.Identity()
        self.fc = nn.Linear(768, output_dim)

    def forward(self, x):
        features = self.model(x)
        return self.fc(features)

class CaptionGenerator(nn.Module):
    def __init__(self, embedding_size, hidden_dim, vocab_size, vocab, num_layers=1):
        super(CaptionGenerator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.end_token = vocab("<end>")

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        lstm_input = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        lstm_output, _ = self.lstm(lstm_input)
        return self.fc(lstm_output)

    def predict(self, features, max_len=20):
        results = []
        input_features = features.unsqueeze(1)
        hidden = None
        for _ in range(max_len):
            lstm_output, hidden = self.lstm(input_features, hidden)
            logits = self.fc(lstm_output.squeeze(1))
            predicted = logits.argmax(dim=1)
            results.append(predicted.item())
            if predicted.item() == self.end_token:
                break
            input_features = self.embed(predicted).unsqueeze(1)
        return results

vocab = Vocabulary(annotations_file='../COCO_Data/annotations/captions_train2017.json', vocab_exists=True)

images_dir = '../COCO_Data/train2017'
captions_path = '../COCO_Data/annotations/captions_train2017.json'

dataloader = get_data_loader(
        images_dir=images_dir,
        captions_path=captions_path,
        vocab_exists=False,
        batch_size=128,
        transform=transform
    )

# Paths and Setup
image_folder = "../coco_dataset/train2017"
caption_file = "../coco_dataset/annotations/captions_train2017.json"
embedding_size = 256
hidden_dim = 512
batch_size = 32
epochs = 10

# Data processing
transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

vocabulary = CaptionVocabulary(caption_file)
train_loader = create_data_loader(image_folder, caption_file, vocabulary, batch_size, transformations)

# Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder_model = TransformerEncoder(embedding_size).to(device)
decoder_model = CaptionGenerator(embedding_size, hidden_dim, len(vocabulary), vocabulary).to(device)

# Optimizer and Loss
criterion = nn.CrossEntropyLoss(ignore_index=0)
encoder_optimizer = optim.Adam(encoder_model.parameters(), lr=0.001)
decoder_optimizer = optim.Adam(decoder_model.parameters(), lr=0.005)

# Training
encoder_model.train()
decoder_model.train()
for epoch in range(epochs):
    epoch_loss = 0
    for data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        images, captions = data["image"].to(device), data["caption"].to(device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        img_features = encoder_model(images)
        outputs = decoder_model(img_features, captions)
        loss = criterion(outputs.view(-1, len(vocabulary)), captions.view(-1))
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}, Avg Loss: {epoch_loss / len(train_loader):.4f}")


    # bleu score
    encoder_model.eval()
    decoder_model.eval()
    references, hypotheses = [], []
    for batch in dataloader:
        images = batch["image"].to(device)
        captions = batch["caption"]
        references.append([[vocab.idx2word[idx] for idx in captions[0].tolist() if idx not in {0, 1, 2}]])
        with torch.no_grad():
            image_features = encoder_model(images)
            generated_ids = decoder_model.generate_caption(image_features)
        hypotheses.append([vocab.idx2word[idx] for idx in generated_ids if idx not in {0, 1, 2}])
    bleu_score = corpus_bleu(references, hypotheses)
    print(f"Epoch {epoch+1}, BLEU Score: {bleu_score:.4f}")
    encoder_model.train()
    decoder_model.train()

torch.save(encoder_model.state_dict(), "custom_encoder.pth")
torch.save(decoder_model.state_dict(), "custom_decoder.pth")

#Evaluation
images_dir_val = '../coco_dataset/val2017'
captions_path_val = '../coco_dataset/annotations/captions_val2017.json'
dataloader_val = get_data_loader(images_dir_val, captions_path_val, vocab, batch_size=1, transform=transform)

references, hypotheses = [], []
for batch in tqdm(dataloader_val, desc="Evaluating", unit="batch"):
    images = batch["image"].to(device)
    captions = batch["caption"]
    references.append([[vocab.idx2word[idx] for idx in captions[0].tolist() if idx not in {0, 1, 2}]])
    with torch.no_grad():
        image_features = encoder_model(images)
        generated_ids = decoder_model.generate_caption(image_features)
    hypotheses.append([vocab.idx2word[idx] for idx in generated_ids if idx not in {0, 1, 2}])
final_bleu_score = corpus_bleu(references, hypotheses)
print(f"Final BLEU Score: {final_bleu_score:.4f}")


