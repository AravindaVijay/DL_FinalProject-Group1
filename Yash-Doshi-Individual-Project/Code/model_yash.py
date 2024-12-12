
######################################## RESNET50 ########################################################

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from PIL import Image
from tqdm import tqdm
import random


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --------------------
# Vocabulary Class
# --------------------
class Vocabulary:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.word_count = 4

    def build_vocab(self, captions):
        for caption in captions:
            for word in caption.split():
                if word not in self.word2idx:
                    self.word2idx[word] = self.word_count
                    self.idx2word[self.word_count] = word
                    self.word_count += 1

    def __len__(self):
        return len(self.word2idx)

    def tokenize(self, caption):
        return [self.word2idx.get(word, self.word2idx["<unk>"]) for word in caption.split()]

    def detokenize(self, tokens):
        return " ".join([self.idx2word[token] for token in tokens if token not in [0, 1, 2]])

# --------------------
# COCO Dataset Class
# --------------------
class CocoDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform, vocab):
        self.img_dir = img_dir
        self.vocab = vocab
        with open(annotations_file, "r") as f:
            data = json.load(f)
        self.annotations = data["annotations"]
        self.image_info = {img["id"]: img["file_name"] for img in data["images"]}
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.img_dir, self.image_info[annotation["image_id"]])
        caption = annotation["caption"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        tokenized_caption = [self.vocab.word2idx["<start>"]] + \
                            self.vocab.tokenize(caption) + \
                            [self.vocab.word2idx["<end>"]]
        return image, torch.tensor(tokenized_caption)

# --------------------
# Collate Function
# --------------------
def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    caption_lengths = [len(cap) for cap in captions]
    max_length = max(caption_lengths)
    padded_captions = torch.zeros((len(captions), max_length), dtype=torch.long)
    for i, cap in enumerate(captions):
        padded_captions[i, :len(cap)] = cap
    return images, padded_captions, torch.tensor(caption_lengths)

# --------------------
# Encoder
# --------------------
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1]  
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

# --------------------
# Decoder
# --------------------
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.fc(hiddens)
        return outputs

    def generate_caption(self, features, vocab, max_len=20):
        generated = []
        inputs = features.unsqueeze(1)
        states = None
        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.fc(hiddens.squeeze(1))
            predicted = outputs.argmax(1)
            word = vocab.idx2word[predicted.item()]
            if word == "<end>":
                break
            generated.append(word)
            inputs = self.embed(predicted).unsqueeze(1)
        return " ".join(generated)

# --------------------
# BLEU Score Calculation
# --------------------
def compute_bleu_score(references, hypotheses):
    def ngram_counts(sequence, n):
        return {tuple(sequence[i:i + n]): 1 for i in range(len(sequence) - n + 1)}

    def precision(ref, hyp, n):
        ref_ngrams = ngram_counts(ref, n)
        hyp_ngrams = ngram_counts(hyp, n)
        matched = sum(1 for ngram in hyp_ngrams if ngram in ref_ngrams)
        return matched / max(1, len(hyp_ngrams))

    precisions = [precision(ref, hyp, n) for n in range(1, 5)]
    brevity_penalty = min(1.0, len(hyp) / len(ref))
    bleu = brevity_penalty * np.exp(sum(np.log(p) if p > 0 else -1e6 for p in precisions) / 4)
    return bleu

# --------------------
# Training Function
# --------------------
def train_model(encoder, decoder, dataloader, criterion, optimizer, device):
    encoder.train()
    decoder.train()
    total_loss = 0

    for images, captions, lengths in tqdm(dataloader, desc="Training"):
        images, captions = images.to(device), captions.to(device)
        optimizer.zero_grad()

        features = encoder(images)
        outputs = decoder(features, captions[:, :-1])  
        target_length = captions[:, 1:].shape[1]
        output_length = outputs.shape[1]

        if output_length > target_length:
            outputs = outputs[:, :target_length, :]
        elif output_length < target_length:
            padding = torch.zeros((outputs.shape[0], target_length - output_length, outputs.shape[2]),
                                   device=outputs.device)
            outputs = torch.cat([outputs, padding], dim=1)

        loss = criterion(outputs.reshape(-1, outputs.size(2)), captions[:, 1:].reshape(-1))  
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

# --------------------
# Main Script
# --------------------
def main():
    train_images = "../COCO_Data/train2017"
    train_captions = "../COCO_Data/annotations/captions_train2017.json"
    val_images = "../COCO_Data/val2017"
    val_captions = "../COCO_Data/annotations/captions_val2017.json"

    embed_size = 256
    hidden_size = 512
    batch_size = 128
    num_epochs = 1
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocabulary()
    with open(train_captions, "r") as f:
        annotations = json.load(f)["annotations"]
    captions = [ann["caption"] for ann in annotations]
    vocab.build_vocab(captions)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_dataset = CocoDataset(train_images, train_captions, transform, vocab)
    val_dataset = CocoDataset(val_images, val_captions, transform, vocab)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<pad>"])
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss = train_model(encoder, decoder, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")

    torch.save(encoder.state_dict(), "model_yash_encoder.pth")
    torch.save(decoder.state_dict(), "model_yash_decoder.pth")
    print("Models saved as 'model_yash_encoder.pth' and 'model_yash_decoder.pth'.")

    encoder.eval()
    decoder.eval()
    references, hypotheses = [], []

    with torch.no_grad():
        for images, captions, lengths in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            features = encoder(images)
            generated_caption = decoder.generate_caption(features, vocab)
            references.append(captions[0].tolist())
            hypotheses.append([vocab.word2idx[word] for word in generated_caption.split()])

    bleu_score = compute_bleu_score(references, hypotheses)
    print(f"BLEU Score: {bleu_score:.4f}")

if __name__ == "__main__":
    main()


