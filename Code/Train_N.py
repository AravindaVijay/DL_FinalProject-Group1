##########################################################################
######################PRETRAINED RESNET AND LSTM########################
##########################################################################

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision.models import resnet50, ResNet50_Weights
# from torchvision import transforms  # Added missing import
# from data_loader import get_data_loader
# from model import ImageEncoder, CaptionDecoder

# # Hyperparameters
# num_epochs = 2  # Define the number of epochs for training
# batch_size = 16
# learning_rate = 0.001
# embedding_dim = 256
# hidden_dim = 512

# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Paths to COCO dataset
# images_dir = '../COCO_Data/train2017'
# captions_path = '../COCO_Data/annotations/captions_train2017.json'

# # Load data
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# dataloader = get_data_loader(images_dir, captions_path, vocab_exists=False, batch_size=batch_size, transform=transform)

# # Initialize models
# encoder = ImageEncoder(embedding_dim=embedding_dim).to(device)
# decoder = CaptionDecoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=len(dataloader.dataset.vocab)).to(device)

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

# # Training loop
# print("Starting training...")
# for epoch in range(num_epochs):
#     encoder.train()
#     decoder.train()

#     for i, batch in enumerate(dataloader):
#         images = batch['images'].to(device)
#         captions = batch['tokenized_caption'].to(device)

#         # Forward pass
#         features = encoder(images)
#         outputs = decoder(features, captions)
        
#         # Compute loss
#         loss = criterion(outputs.view(-1, len(dataloader.dataset.vocab)), captions.view(-1))

#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # Print progress
#         if i % 10 == 0:
#             print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

#     # Save the model after each epoch
#     torch.save(encoder.state_dict(), f"encoder_epoch_{epoch+1}.pth")
#     torch.save(decoder.state_dict(), f"decoder_epoch_{epoch+1}.pth")
#     print(f"Epoch {epoch+1}/{num_epochs} completed. Models saved.")

# print("Training completed successfully!")

##########################################################################
###################### CNN AND RNN ########################
##########################################################################

import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision.models import resnet50, ResNet50_Weights
# from torchvision import transforms  # Added missing import
# from data_loader import get_data_loader
# # from model import ImageEncoder, CaptionDecoder
# from model import CustomCNNEncoder, CustomRNNDecoder

# # # Hyperparameters
# num_epochs = 2  # Define the number of epochs for training
# batch_size = 16
# # learning_rate = 0.001
# embedding_dim = 256
# hidden_dim = 512

# # Initialize custom models
# # embedding_dim = 256
# # hidden_dim = 512

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Paths to COCO dataset
# images_dir = '../COCO_Data/train2017'
# captions_path = '../COCO_Data/annotations/captions_train2017.json'

# # Load data
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# dataloader = get_data_loader(images_dir, captions_path, vocab_exists=False, batch_size=batch_size, transform=transform)

# # # Initialize models
# # encoder = ImageEncoder(embedding_dim=embedding_dim).to(device)
# # decoder = CaptionDecoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=len(dataloader.dataset.vocab)).to(device)

# encoder = CustomCNNEncoder(embedding_dim=embedding_dim).to(device)
# decoder = CustomRNNDecoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=len(dataloader.dataset.vocab)).to(device)

# # Set device
# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

# # Training loop
# print("Starting training...")
# for epoch in range(num_epochs):
#     encoder.train()
#     decoder.train()

#     for i, batch in enumerate(dataloader):
#         images = batch['images'].to(device)
#         captions = batch['tokenized_caption'].to(device)

#         # Forward pass
#         features = encoder(images)
#         outputs = decoder(features, captions)
        
#         # Compute loss
#         loss = criterion(outputs.view(-1, len(dataloader.dataset.vocab)), captions.view(-1))

#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # Print progress
#         if i % 10 == 0:
#             print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

#     # Save the model after each epoch
#     torch.save(encoder.state_dict(), f"encoder_epoch_{epoch+1}.pth")
#     torch.save(decoder.state_dict(), f"decoder_epoch_{epoch+1}.pth")
#     print(f"Epoch {epoch+1}/{num_epochs} completed. Models saved.")

# print("Training completed successfully!")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from tqdm import tqdm
import timm
from pycocotools.coco import COCO
import random
import numpy as np
import nltk
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import corpus_bleu  # Added for BLEU score computation


# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Vocabulary Class
class Vocabulary:
    def __init__(self, annotations_file, vocab_threshold=5):
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.word_count = {}
        self.vocab_threshold = vocab_threshold
        self.build_vocab(annotations_file)

    def build_vocab(self, annotations_file):
        coco = COCO(annotations_file)
        for ann_id in coco.anns.keys():
            caption = coco.anns[ann_id]["caption"]
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            for token in tokens:
                self.word_count[token] = self.word_count.get(token, 0) + 1
        for word, count in self.word_count.items():
            if count >= self.vocab_threshold:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx["<unk>"])

    def __len__(self):
        return len(self.word2idx)

# Dataset Class
class COCOCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, captions_path, vocab, transform=None):
        self.coco = COCO(captions_path)
        self.image_ids = list(self.coco.imgs.keys())
        self.images_dir = images_dir
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = f"{self.images_dir}/{image_info['file_name']}"
        # Load and transform the image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            raise ValueError("Transformations must be provided to resize images consistently.")
        # Load and process the caption
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        captions = [self.coco.anns[ann_id]["caption"] for ann_id in ann_ids]
        tokenized_caption = [self.vocab("<start>")] + \
                            [self.vocab(token) for token in nltk.tokenize.word_tokenize(captions[0].lower())] + \
                            [self.vocab("<end>")]
        tokenized_caption = torch.tensor(tokenized_caption, dtype=torch.long)
        return {"image": image, "caption": tokenized_caption}

# DataLoader Function
def get_data_loader(images_dir, captions_path, vocab, batch_size, transform):
    dataset = COCOCaptionDataset(images_dir, captions_path, vocab, transform)
    def collate_fn(batch):
        images = torch.stack([item["image"] for item in batch])
        captions = [item["caption"] for item in batch]
        captions_padded = pad_sequence(captions, batch_first=True, padding_value=0)
        return {"image": images, "caption": captions_padded}
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Vision Transformer Encoder
class VisionTransformerEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(VisionTransformerEncoder, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()
        self.fc = nn.Linear(768, embedding_dim)

    def forward(self, images):
        features = self.vit(images)
        embeddings = self.fc(features)
        return embeddings

# Caption Decoder
class CaptionDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, vocab, num_layers=1):
        super(CaptionDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.end_token_id = vocab("<end>")

    def forward(self, image_features, captions):
        embeddings = self.word_embeddings(captions[:, :-1])
        inputs = torch.cat((image_features.unsqueeze(1), embeddings), dim=1)
        lstm_output, _ = self.lstm(inputs)
        outputs = self.fc(lstm_output)
        return outputs

    def generate_caption(self, image_features, max_length=20):
        generated_ids = []
        states = None
        inputs = image_features.unsqueeze(1)
        for _ in range(max_length):
            lstm_output, states = self.lstm(inputs, states)
            outputs = self.fc(lstm_output.squeeze(1))
            predicted_token = outputs.argmax(dim=1)
            generated_ids.append(predicted_token.item())
            if predicted_token.item() == self.end_token_id:
                break
            inputs = self.word_embeddings(predicted_token).unsqueeze(1)
        return generated_ids

# Paths and Initialization
images_dir = '../coco_dataset/train2017'
captions_path = '../coco_dataset/annotations/captions_train2017.json'
embedding_dim = 256
hidden_dim = 512
batch_size = 64
num_epochs = 1

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

vocab = Vocabulary(annotations_file=captions_path, vocab_threshold=5)
dataloader = get_data_loader(images_dir, captions_path, vocab, batch_size, transform)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = VisionTransformerEncoder(embedding_dim).to(device)
decoder = CaptionDecoder(embedding_dim, hidden_dim, len(vocab), vocab).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<pad>"])
encoder_optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=5e-3)

# Training Loop
encoder.train()
decoder.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
        images = batch["image"].to(device)
        captions = batch["caption"].to(device)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        image_features = encoder(images)
        outputs = decoder(image_features, captions)
        loss = criterion(outputs.view(-1, len(vocab)), captions.view(-1))
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

    # Compute BLEU Score after each epoch
    encoder.eval()
    decoder.eval()
    references, hypotheses = [], []
    for batch in dataloader:
        images = batch["image"].to(device)
        captions = batch["caption"]
        references.append([[vocab.idx2word[idx] for idx in captions[0].tolist() if idx not in {0, 1, 2}]])
        with torch.no_grad():
            image_features = encoder(images)
            generated_ids = decoder.generate_caption(image_features)
        hypotheses.append([vocab.idx2word[idx] for idx in generated_ids if idx not in {0, 1, 2}])
    bleu_score = corpus_bleu(references, hypotheses)
    print(f"Epoch {epoch+1}, BLEU Score: {bleu_score:.4f}")
    encoder.train()
    decoder.train()

torch.save(encoder.state_dict(), 'encoder_vit.pth')
torch.save(decoder.state_dict(), 'decoder.pth')

# Final Evaluation
images_dir_val = '../coco_dataset/val2017'
captions_path_val = '../coco_dataset/annotations/captions_val2017.json'
dataloader_val = get_data_loader(images_dir_val, captions_path_val, vocab, batch_size=1, transform=transform)

references, hypotheses = [], []
for batch in tqdm(dataloader_val, desc="Evaluating", unit="batch"):
    images = batch["image"].to(device)
    captions = batch["caption"]
    references.append([[vocab.idx2word[idx] for idx in captions[0].tolist() if idx not in {0, 1, 2}]])
    with torch.no_grad():
        image_features = encoder(images)
        generated_ids = decoder.generate_caption(image_features)
    hypotheses.append([vocab.idx2word[idx] for idx in generated_ids if idx not in {0, 1, 2}])
final_bleu_score = corpus_bleu(references, hypotheses)
print(f"Final BLEU Score: {final_bleu_score:.4f}")