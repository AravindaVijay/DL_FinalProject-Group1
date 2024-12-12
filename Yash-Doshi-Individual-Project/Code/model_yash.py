# import tensorflow as tf
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Embedding, LSTM, Add, Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.preprocessing.text import Tokenizer
# import tensorflow_datasets as tfds
# import numpy as np
# import matplotlib.pyplot as plt

# # -------------------------------
# # Step 1: Load and Preprocess Dataset
# # -------------------------------

# # Load Flickr8k dataset
# data, info = tfds.load('flickr8k', with_info=True, as_supervised=True)
# train_data = data['train']
# test_data = data['test']

# # Collect captions and images
# captions = []
# images = []

# for image, caption in train_data:
#     captions.append(caption.numpy().decode('utf-8'))
#     images.append(image)

# # Tokenize captions
# tokenizer = Tokenizer(num_words=5000, oov_token="<unk>")
# tokenizer.fit_on_texts(captions)
# vocab_size = len(tokenizer.word_index) + 1  # Include padding token
# print(f"Vocabulary Size: {vocab_size}")

# # Convert captions to sequences
# tokenized_captions = tokenizer.texts_to_sequences(captions)

# # Pad captions
# max_caption_length = 20
# padded_captions = pad_sequences(tokenized_captions, maxlen=max_caption_length, padding='post')

# # -------------------------------
# # Step 2: Define Custom CNN
# # -------------------------------

# def build_cnn():
#     inputs = Input(shape=(224, 224, 3))
#     x = Conv2D(32, (3, 3), activation='relu')(inputs)
#     x = MaxPooling2D((2, 2))(x)
#     x = Conv2D(64, (3, 3), activation='relu')(x)
#     x = MaxPooling2D((2, 2))(x)
#     x = Conv2D(128, (3, 3), activation='relu')(x)
#     x = MaxPooling2D((2, 2))(x)
#     x = Flatten()(x)
#     x = Dense(256, activation='relu')(x)
#     return tf.keras.Model(inputs, x)

# cnn = build_cnn()
# cnn.summary()

# # -------------------------------
# # Step 3: Define Custom RNN
# # -------------------------------

# def build_rnn(vocab_size, max_caption_length):
#     inputs = Input(shape=(max_caption_length,))
#     x = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(inputs)
#     x = LSTM(256, return_sequences=False)(x)
#     x = Dense(256, activation='relu')(x)
#     return tf.keras.Model(inputs, x)

# rnn = build_rnn(vocab_size=vocab_size, max_caption_length=max_caption_length)
# rnn.summary()

# # -------------------------------
# # Step 4: Combine CNN and RNN
# # -------------------------------

# def build_image_captioning_model(vocab_size, max_caption_length):
#     # Image input and features
#     image_input = cnn.input
#     image_features = cnn.output

#     # Caption input and features
#     caption_input = rnn.input
#     caption_features = rnn.output

#     # Combine image and caption features
#     combined = Add()([image_features, caption_features])
#     output = Dense(vocab_size, activation='softmax')(combined)

#     # Define model
#     return Model(inputs=[image_input, caption_input], outputs=output)

# model = build_image_captioning_model(vocab_size, max_caption_length)
# model.compile(optimizer='adam', loss='categorical_crossentropy')
# model.summary()

# # -------------------------------
# # Step 5: Prepare Data for Training
# # -------------------------------

# def preprocess_image(image):
#     image = tf.image.resize(image, (224, 224)) / 255.0  # Normalize
#     return image

# def create_sequences(tokenized_captions, images, max_caption_length):
#     X1, X2, y = [], [], []
#     for i, caption in enumerate(tokenized_captions):
#         for t in range(1, len(caption)):
#             # Input sequence
#             in_seq = caption[:t]
#             # Output word
#             out_seq = caption[t]
#             # Pad input sequence
#             in_seq = pad_sequences([in_seq], maxlen=max_caption_length, padding='post')[0]
#             # Preprocess image
#             image = preprocess_image(images[i])
#             features = cnn(tf.expand_dims(image, axis=0))
#             # Append data
#             X1.append(features.numpy().flatten())
#             X2.append(in_seq)
#             y.append(out_seq)
#     return np.array(X1), np.array(X2), np.array(y)

# # Prepare training data
# X1, X2, y = create_sequences(tokenized_captions, images, max_caption_length)
# y = to_categorical(y, num_classes=vocab_size)
# print(f"Training Data Shapes: X1={X1.shape}, X2={X2.shape}, y={y.shape}")

# # -------------------------------
# # Step 6: Train the Model
# # -------------------------------

# model.fit([X1, X2], y, batch_size=32, epochs=10)

# # -------------------------------
# # Step 7: Generate Captions
# # -------------------------------

# def generate_caption(model, tokenizer, image, max_caption_length):
#     # Preprocess the image
#     image = preprocess_image(image)
#     features = cnn(tf.expand_dims(image, axis=0)).numpy().flatten().reshape(1, -1)
#     # Start generating caption
#     input_seq = [tokenizer.word_index['<start>']]
#     for _ in range(max_caption_length):
#         # Pad sequence
#         sequence = pad_sequences([input_seq], maxlen=max_caption_length, padding='post')
#         # Predict next word
#         yhat = model.predict([features, sequence], verbose=0)
#         word_index = np.argmax(yhat)
#         word = tokenizer.index_word.get(word_index, '<unk>')
#         if word == '<end>':
#             break
#         input_seq.append(word_index)
#     return ' '.join([tokenizer.index_word[i] for i in input_seq[1:]])

# # -------------------------------
# # Step 8: Test the Model
# # -------------------------------

# test_image, _ = next(iter(test_data))  # Load a test image
# plt.imshow(test_image.numpy())
# plt.axis('off')
# plt.show()

# # Generate caption
# # caption = generate_caption(model, tokenizer, test_image, max_caption_length)
# # print("Generated Caption:", caption)
# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# from torchvision import models
# import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2
# from tqdm import tqdm
# from pycocotools.coco import COCO
# from nltk.translate.bleu_score import corpus_bleu
# from data_loader import get_data_loader
# from vocabulary import Vocabulary
# from torch.cuda.amp import GradScaler, autocast
# import numpy as np

# data, info = tfds.load('coco_captions', with_info=True)
# train_data = data['train']
# test_data = data['test']

# captions = []
# images = []

# for example in train_data:
#     image = example['image']
#     caption_list = example['captions']
#     for caption in caption_list:
#         captions.append(caption)
#         images.append(image.numpy())

# print(f"Number of samples: {len(captions)}")
# print("Sample Caption:", captions[0])
# plt.imshow(images[0])
# plt.axis('off')
# plt.show()

# tokenizer = Tokenizer(num_words=5000, oov_token="<unk>")
# tokenizer.fit_on_texts(captions)
# vocab_size = len(tokenizer.word_index) + 1

# tokenized_captions = tokenizer.texts_to_sequences(captions)
# max_caption_length = 20
# padded_captions = pad_sequences(tokenized_captions, maxlen=max_caption_length, padding='post')

# def build_cnn():
#     inputs = Input(shape=(224, 224, 3))
#     x = Conv2D(32, (3, 3), activation='relu')(inputs)
#     x = MaxPooling2D((2, 2))(x)
#     x = Conv2D(64, (3, 3), activation='relu')(x)
#     x = MaxPooling2D((2, 2))(x)
#     x = Conv2D(128, (3, 3), activation='relu')(x)
#     x = MaxPooling2D((2, 2))(x)
#     x = Flatten()(x)
#     x = Dense(256, activation='relu')(x)
#     return tf.keras.Model(inputs, x)

#     def forward(self, images):
#         features = self.features(images)  # Extract feature maps
#         features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)  # Flatten
#         return self.fc(features)

# def build_rnn(vocab_size, max_caption_length):
#     inputs = Input(shape=(max_caption_length,))
#     x = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(inputs)
#     x = LSTM(256, return_sequences=False)(x)
#     x = Dense(256, activation='relu')(x)
#     return tf.keras.Model(inputs, x)

#         self.attention = Attention(embedding_dim, hidden_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, vocab_size)

# def build_image_captioning_model(vocab_size, max_caption_length):
#     image_input = cnn.input
#     image_features = cnn.output
#     caption_input = rnn.input
#     caption_features = rnn.output
#     combined = Add()([image_features, caption_features])
#     output = Dense(vocab_size, activation='softmax')(combined)
#     return Model(inputs=[image_input, caption_input], outputs=output)

# train_loader = get_data_loader(train_images_folder, captions_file, vocab_exists=True, batch_size=64, transform=train_transform)
# val_loader = get_data_loader(val_images_folder, val_captions_file, vocab_exists=True, batch_size=1, transform=val_transform)

# def preprocess_image(image):
#     image = tf.image.resize(image, (224, 224)) / 255.0
#     return image

# def create_sequences(tokenized_captions, images, max_caption_length):
#     X1, X2, y = [], [], []
#     for i, caption in enumerate(tokenized_captions):
#         for t in range(1, len(caption)):
#             in_seq = caption[:t]
#             out_seq = caption[t]
#             in_seq = pad_sequences([in_seq], maxlen=max_caption_length, padding='post')[0]
#             image = preprocess_image(images[i])
#             features = cnn(tf.expand_dims(image, axis=0))
#             X1.append(features.numpy().flatten())
#             X2.append(in_seq)
#             y.append(out_seq)
#     return np.array(X1), np.array(X2), np.array(y)

# num_samples = 1000
# X1, X2, y = create_sequences(tokenized_captions[:num_samples], images[:num_samples], max_caption_length)
# y = to_categorical(y, num_classes=vocab_size)

#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         total_loss += loss.item()

# def generate_caption(model, tokenizer, image, max_caption_length):
#     image = preprocess_image(image)
#     features = cnn(tf.expand_dims(image, axis=0)).numpy().flatten().reshape(1, -1)
#     input_seq = [tokenizer.word_index['<start>']]
#     for _ in range(max_caption_length):
#         sequence = pad_sequences([input_seq], maxlen=max_caption_length, padding='post')
#         yhat = model.predict([features, sequence], verbose=0)
#         word_index = np.argmax(yhat)
#         word = tokenizer.index_word.get(word_index, '<unk>')
#         if word == '<end>':
#             break
#         input_seq.append(word_index)
#     return ' '.join([tokenizer.index_word[i] for i in input_seq[1:]])

# test_image, _ = next(iter(test_data))
# plt.imshow(test_image.numpy())
# plt.axis('off')
# plt.show()

# caption = generate_caption(model, tokenizer, test_image, max_caption_length)
# print("Generated Caption:", caption)



### DIFFERENT CODE

# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# from torchvision import models
# import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2
# from tqdm import tqdm
# from pycocotools.coco import COCO
# from nltk.translate.bleu_score import corpus_bleu
# from data_loader import get_data_loader
# from vocabulary import Vocabulary
# from torch.cuda.amp import GradScaler, autocast

# # Set random seed for reproducibility
# import random
# import numpy as np
# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)

# # Paths
# annotations_folder = "../COCO_Data/annotations"
# train_images_folder = "../COCO_Data/train2017"
# val_images_folder = "../COCO_Data/val2017"
# captions_file = os.path.join(annotations_folder, "captions_train2017.json")
# val_captions_file = os.path.join(annotations_folder, "captions_val2017.json")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Vocabulary
# vocab = Vocabulary(annotations_file=captions_file, vocab_exists=True)

# # Image Augmentation
# train_transform = A.Compose([
#     A.Resize(256, 256),
#     A.RandomCrop(224, 224),
#     A.HorizontalFlip(p=0.5),
#     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ToTensorV2()
# ])
# val_transform = A.Compose([
#     A.Resize(224, 224),
#     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ToTensorV2()
# ])

# # Attention Mechanism
# class Attention(nn.Module):
#     def __init__(self, feature_dim, hidden_dim):
#         super(Attention, self).__init__()
#         self.attention = nn.Linear(feature_dim + hidden_dim, hidden_dim)
#         self.v = nn.Linear(hidden_dim, 1)

#     def forward(self, image_features, hidden_state):
#         hidden_state = hidden_state.unsqueeze(1).repeat(1, image_features.size(1), 1)
#         combined_features = torch.cat((image_features, hidden_state), dim=2)
#         attention_weights = torch.softmax(self.v(torch.tanh(self.attention(combined_features))), dim=1)
#         attended_features = (attention_weights * image_features).sum(dim=1)
#         return attended_features, attention_weights

# # Encoder using ResNet-101
# class ResNet101Encoder(nn.Module):
#     def __init__(self, embedding_dim):
#         super(ResNet101Encoder, self).__init__()
#         resnet = models.resnet101(pretrained=True)
#         self.features = nn.Sequential(*list(resnet.children())[:-2])  # Use all layers except the classifier
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(2048, embedding_dim)

#         # Freeze the ResNet layers
#         for param in self.features.parameters():
#             param.requires_grad = False

#     def forward(self, images):
#         features = self.features(images)  # Extract feature maps
#         features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)  # Flatten
#         return self.fc(features)

# # Decoder with Attention
# class CaptionDecoder(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers=1):
#         super(CaptionDecoder, self).__init__()
#         self.attention = Attention(embedding_dim, hidden_dim)
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, vocab_size)

#     def forward(self, image_features, captions):
#         embeddings = self.embedding(captions[:, :-1])
#         h, c = torch.zeros((1, image_features.size(0), hidden_dim)).to(device), torch.zeros((1, image_features.size(0), hidden_dim)).to(device)
#         outputs = []
#         for t in range(embeddings.size(1)):
#             attended_features, _ = self.attention(image_features, h[-1])
#             lstm_input = torch.cat((attended_features.unsqueeze(1), embeddings[:, t].unsqueeze(1)), dim=2)
#             output, (h, c) = self.lstm(lstm_input, (h, c))
#             outputs.append(self.fc(output.squeeze(1)))
#         return torch.stack(outputs, dim=1)

# # Training Setup
# embedding_dim = 512
# hidden_dim = 512
# vocab_size = len(vocab)
# encoder = ResNet101Encoder(embedding_dim).to(device)
# decoder = CaptionDecoder(embedding_dim, hidden_dim, vocab_size).to(device)

# criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<pad>'])
# optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# train_loader = get_data_loader(train_images_folder, captions_file, vocab_exists=True, batch_size=64, transform=train_transform)
# val_loader = get_data_loader(val_images_folder, val_captions_file, vocab_exists=True, batch_size=1, transform=val_transform)

# # Mixed Precision Training
# scaler = GradScaler()

# # Training Loop
# num_epochs = 1
# for epoch in range(num_epochs):
#     encoder.train()
#     decoder.train()
#     total_loss = 0

#     for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
#         images = batch['images'].to(device)
#         captions = batch['tokenized_caption'].to(device)

#         optimizer.zero_grad()

#         with autocast():
#             image_features = encoder(images)
#             outputs = decoder(image_features, captions)
#             loss = criterion(outputs.view(-1, vocab_size), captions[:, 1:].reshape(-1))

#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         total_loss += loss.item()

#     avg_loss = total_loss / len(train_loader)
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
#     scheduler.step()

# torch.save(encoder.state_dict(), "encoder_resnet101_attention.pth")
# torch.save(decoder.state_dict(), "decoder_resnet101_attention.pth")

# # Evaluation
# def evaluate_model(encoder, decoder, val_loader, max_length=20):
#     encoder.eval()
#     decoder.eval()
#     references, hypotheses = [], []

#     for batch in tqdm(val_loader, desc="Evaluating"):
#         images = batch['images'].to(device)
#         all_captions = batch['all_captions']

#         with torch.no_grad():
#             image_features = encoder(images)
#             generated_caption = decoder.generate_caption(image_features, max_length=max_length)

#         hypotheses.append(generated_caption.split())
#         references.append([caption.split() for caption in all_captions])

#     bleu_score = corpus_bleu(references, hypotheses)
#     return bleu_score

# encoder.load_state_dict(torch.load("encoder_resnet101_attention.pth"))
# decoder.load_state_dict(torch.load("decoder_resnet101_attention.pth"))

# bleu = evaluate_model(encoder, decoder, val_loader)
# print(f"BLEU Score: {bleu:.4f}")


#########




# import os
# import json
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, models
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# import random

# # Set random seed for reproducibility
# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)

# # --------------------
# # Vocabulary Class
# # --------------------
# class Vocabulary:
#     def __init__(self):
#         self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
#         self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
#         self.word_count = 4

#     def build_vocab(self, captions):
#         for caption in captions:
#             for word in caption.split():
#                 if word not in self.word2idx:
#                     self.word2idx[word] = self.word_count
#                     self.idx2word[self.word_count] = word
#                     self.word_count += 1

#     def __len__(self):
#         return len(self.word2idx)

#     def tokenize(self, caption):
#         return [self.word2idx.get(word, self.word2idx["<unk>"]) for word in caption.split()]

#     def detokenize(self, tokens):
#         return " ".join([self.idx2word[token] for token in tokens if token not in [0, 1, 2]])

# # --------------------
# # COCO Dataset Class
# # --------------------
# class CocoDataset(Dataset):
#     def __init__(self, img_dir, annotations_file, transform, vocab):
#         self.img_dir = img_dir
#         self.vocab = vocab
#         with open(annotations_file, "r") as f:
#             data = json.load(f)
#         self.annotations = data["annotations"]
#         self.image_info = {img["id"]: img["file_name"] for img in data["images"]}
#         self.transform = transform

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, idx):
#         annotation = self.annotations[idx]
#         img_path = os.path.join(self.img_dir, self.image_info[annotation["image_id"]])
#         caption = annotation["caption"]
#         image = Image.open(img_path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#         tokenized_caption = [self.vocab.word2idx["<start>"]] + \
#                             self.vocab.tokenize(caption) + \
#                             [self.vocab.word2idx["<end>"]]
#         return image, torch.tensor(tokenized_caption)

# # --------------------
# # Collate Function
# # --------------------
# def collate_fn(batch):
#     images, captions = zip(*batch)
#     images = torch.stack(images, dim=0)
#     caption_lengths = [len(cap) for cap in captions]
#     max_length = max(caption_lengths)
#     padded_captions = torch.zeros((len(captions), max_length), dtype=torch.long)
#     for i, cap in enumerate(captions):
#         padded_captions[i, :len(cap)] = cap
#     return images, padded_captions, torch.tensor(caption_lengths)

# # --------------------
# # Encoder
# # --------------------
# class EncoderCNN(nn.Module):
#     def __init__(self, embed_size):
#         super(EncoderCNN, self).__init__()
#         resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
#         modules = list(resnet.children())[:-1]  # Remove the last FC layer
#         self.resnet = nn.Sequential(*modules)
#         self.embed = nn.Linear(resnet.fc.in_features, embed_size)

#     def forward(self, images):
#         features = self.resnet(images)
#         features = features.view(features.size(0), -1)
#         features = self.embed(features)
#         return features

# # --------------------
# # Decoder
# # --------------------
# class DecoderRNN(nn.Module):
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
#         super(DecoderRNN, self).__init__()
#         self.embed = nn.Embedding(vocab_size, embed_size)
#         self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, vocab_size)

#     def forward(self, features, captions):
#         embeddings = self.embed(captions)
#         embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
#         hiddens, _ = self.lstm(embeddings)
#         outputs = self.fc(hiddens)
#         return outputs

# # --------------------
# # Training Function
# # --------------------
# def train_model(encoder, decoder, dataloader, criterion, optimizer, device):
#     encoder.train()
#     decoder.train()
#     total_loss = 0

#     for images, captions, lengths in tqdm(dataloader, desc="Training"):
#         images, captions = images.to(device), captions.to(device)
#         optimizer.zero_grad()

#         # Forward pass
#         features = encoder(images)
#         outputs = decoder(features, captions[:, :-1])  # Exclude <end> token from input

#         # Debug shapes
#         print(f"Outputs shape: {outputs.shape}")  # [batch_size, seq_length-1, vocab_size]
#         print(f"Target captions shape: {captions[:, 1:].shape}")  # [batch_size, seq_length-1]

#         # Adjust output length to match target captions
#         target_length = captions[:, 1:].shape[1]
#         if outputs.shape[1] > target_length:
#             outputs = outputs[:, :target_length, :]  # Truncate outputs if longer
#         elif outputs.shape[1] < target_length:
#             padding = torch.zeros((outputs.shape[0], target_length - outputs.shape[1], outputs.shape[2]),
#                                    device=outputs.device)
#             outputs = torch.cat([outputs, padding], dim=1)  # Pad outputs if shorter

#         # Compute loss
#         loss = criterion(outputs.reshape(-1, outputs.size(2)), captions[:, 1:].reshape(-1))  # Exclude <start> token from target
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     return total_loss / len(dataloader)


# # --------------------
# # Main Script
# # --------------------
# def main():
#     # Paths
#     train_images = "../COCO_Data/train2017"
#     train_captions = "../COCO_Data/annotations/captions_train2017.json"

#     # Hyperparameters
#     embed_size = 256
#     hidden_size = 512
#     batch_size = 64  # Adjust batch size for your GPU
#     num_epochs = 1
#     learning_rate = 1e-3
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Build Vocabulary
#     vocab = Vocabulary()
#     with open(train_captions, "r") as f:
#         annotations = json.load(f)["annotations"]
#     captions = [ann["caption"] for ann in annotations]
#     vocab.build_vocab(captions)

#     # Transforms
#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),  # Reduced image size
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ])

#     # Datasets and Dataloaders
#     train_dataset = CocoDataset(train_images, train_captions, transform, vocab)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)

#     # Models, Loss, and Optimizer
#     encoder = EncoderCNN(embed_size).to(device)
#     decoder = DecoderRNN(embed_size, hidden_size, len(vocab)).to(device)
#     criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<pad>"])
#     optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

#     # Training Loop
#     for epoch in range(num_epochs):
#         train_loss = train_model(encoder, decoder, train_loader, criterion, optimizer, device)
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")

# if __name__ == "__main__":
#     main()


# import os
# import json
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, models
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# import random

# # Set random seed for reproducibility
# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)

# # --------------------
# # Vocabulary Class
# # --------------------
# class Vocabulary:
#     def __init__(self):
#         self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
#         self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
#         self.word_count = 4

#     def build_vocab(self, captions):
#         for caption in captions:
#             for word in caption.split():
#                 if word not in self.word2idx:
#                     self.word2idx[word] = self.word_count
#                     self.idx2word[self.word_count] = word
#                     self.word_count += 1

#     def __len__(self):
#         return len(self.word2idx)

#     def tokenize(self, caption):
#         return [self.word2idx.get(word, self.word2idx["<unk>"]) for word in caption.split()]

#     def detokenize(self, tokens):
#         return " ".join([self.idx2word[token] for token in tokens if token not in [0, 1, 2]])

# # --------------------
# # COCO Dataset Class
# # --------------------
# class CocoDataset(Dataset):
#     def __init__(self, img_dir, annotations_file, transform, vocab):
#         self.img_dir = img_dir
#         self.vocab = vocab
#         with open(annotations_file, "r") as f:
#             data = json.load(f)
#         self.annotations = data["annotations"]
#         self.image_info = {img["id"]: img["file_name"] for img in data["images"]}
#         self.transform = transform

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, idx):
#         annotation = self.annotations[idx]
#         img_path = os.path.join(self.img_dir, self.image_info[annotation["image_id"]])
#         caption = annotation["caption"]
#         image = Image.open(img_path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#         tokenized_caption = [self.vocab.word2idx["<start>"]] + \
#                             self.vocab.tokenize(caption) + \
#                             [self.vocab.word2idx["<end>"]]
#         return image, torch.tensor(tokenized_caption)

# # --------------------
# # Collate Function
# # --------------------
# def collate_fn(batch):
#     images, captions = zip(*batch)
#     images = torch.stack(images, dim=0)
#     caption_lengths = [len(cap) for cap in captions]
#     max_length = max(caption_lengths)
#     padded_captions = torch.zeros((len(captions), max_length), dtype=torch.long)
#     for i, cap in enumerate(captions):
#         padded_captions[i, :len(cap)] = cap
#     return images, padded_captions, torch.tensor(caption_lengths)

# # --------------------
# # Encoder
# # --------------------
# class EncoderCNN(nn.Module):
#     def __init__(self, embed_size):
#         super(EncoderCNN, self).__init__()
#         resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
#         modules = list(resnet.children())[:-1]  # Remove the last FC layer
#         self.resnet = nn.Sequential(*modules)
#         self.embed = nn.Linear(resnet.fc.in_features, embed_size)

#     def forward(self, images):
#         features = self.resnet(images)
#         features = features.view(features.size(0), -1)
#         features = self.embed(features)
#         return features

# # --------------------
# # Decoder
# # --------------------
# class DecoderRNN(nn.Module):
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
#         super(DecoderRNN, self).__init__()
#         self.embed = nn.Embedding(vocab_size, embed_size)
#         self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, vocab_size)

#     def forward(self, features, captions):
#         embeddings = self.embed(captions)
#         embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
#         hiddens, _ = self.lstm(embeddings)
#         outputs = self.fc(hiddens)
#         return outputs

#     def generate_caption(self, features, vocab, max_len=20):
#         generated = []
#         inputs = features.unsqueeze(1)
#         states = None
#         for _ in range(max_len):
#             hiddens, states = self.lstm(inputs, states)
#             outputs = self.fc(hiddens.squeeze(1))
#             predicted = outputs.argmax(1)
#             word = vocab.idx2word[predicted.item()]
#             if word == "<end>":
#                 break
#             generated.append(word)
#             inputs = self.embed(predicted).unsqueeze(1)
#         return " ".join(generated)

# # --------------------
# # BLEU Score Calculation
# # --------------------
# def compute_bleu_score(references, hypotheses):
#     """
#     Compute a simple BLEU score.
#     """
#     def ngram_counts(sequence, n):
#         return {tuple(sequence[i:i+n]): 1 for i in range(len(sequence)-n+1)}

#     def precision(ref, hyp, n):
#         ref_ngrams = ngram_counts(ref, n)
#         hyp_ngrams = ngram_counts(hyp, n)
#         matched = sum(1 for ngram in hyp_ngrams if ngram in ref_ngrams)
#         return matched / max(1, len(hyp_ngrams))

#     precisions = [precision(ref, hyp, n) for n in range(1, 5)]
#     bleu = np.exp(sum(np.log(p) if p > 0 else -1e6 for p in precisions) / 4)  # Smoothing for 0 values
#     return bleu

# # --------------------
# # Training Function
# # # --------------------
# # def train_model(encoder, decoder, dataloader, criterion, optimizer, device):
# #     encoder.train()
# #     decoder.train()
# #     total_loss = 0

# #     for images, captions, lengths in tqdm(dataloader, desc="Training"):
# #         images, captions = images.to(device), captions.to(device)
# #         optimizer.zero_grad()

# #         features = encoder(images)
# #         outputs = decoder(features, captions[:, :-1])  # Exclude <end> token from input
# #         loss = criterion(outputs.reshape(-1, outputs.size(2)), captions[:, 1:].reshape(-1))  # Exclude <start> token from target
# #         loss.backward()
# #         optimizer.step()
# #         total_loss += loss.item()

# #     return total_loss / len(dataloader)

# def train_model(encoder, decoder, dataloader, criterion, optimizer, device):
#     encoder.train()
#     decoder.train()
#     total_loss = 0

#     for images, captions, lengths in tqdm(dataloader, desc="Training"):
#         images, captions = images.to(device), captions.to(device)
#         optimizer.zero_grad()

#         # Forward pass
#         features = encoder(images)
#         outputs = decoder(features, captions[:, :-1])  # Exclude <end> token from input

#         # Adjust output length to match target captions
#         max_target_length = captions[:, 1:].shape[1]  # Target excludes <start>
#         max_output_length = outputs.shape[1]

#         if max_output_length > max_target_length:
#             outputs = outputs[:, :max_target_length, :]  # Truncate
#         elif max_output_length < max_target_length:
#             padding = torch.zeros(
#                 (outputs.shape[0], max_target_length - max_output_length, outputs.shape[2]),
#                 device=outputs.device
#             )
#             outputs = torch.cat([outputs, padding], dim=1)  # Pad

#         # Compute loss
#         loss = criterion(outputs.reshape(-1, outputs.size(2)), captions[:, 1:].reshape(-1))  # Exclude <start> token
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     return total_loss / len(dataloader)


# # --------------------
# # Main Script
# # --------------------
# def main():
#     # Paths
#     train_images = "../COCO_Data/train2017"
#     train_captions = "../COCO_Data/annotations/captions_train2017.json"
#     val_images = "../COCO_Data/val2017"
#     val_captions = "../COCO_Data/annotations/captions_val2017.json"

#     # Hyperparameters
#     embed_size = 256
#     hidden_size = 512
#     batch_size = 64
#     num_epochs = 1
#     learning_rate = 1e-3
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Build Vocabulary
#     vocab = Vocabulary()
#     with open(train_captions, "r") as f:
#         annotations = json.load(f)["annotations"]
#     captions = [ann["caption"] for ann in annotations]
#     vocab.build_vocab(captions)

#     # Transforms
#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ])

#     # Datasets and Dataloaders
#     train_dataset = CocoDataset(train_images, train_captions, transform, vocab)
#     val_dataset = CocoDataset(val_images, val_captions, transform, vocab)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
#     val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

#     # Models, Loss, and Optimizer
#     encoder = EncoderCNN(embed_size).to(device)
#     decoder = DecoderRNN(embed_size, hidden_size, len(vocab)).to(device)
#     criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<pad>"])
#     optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

#     # Training Loop
#     for epoch in range(num_epochs):
#         train_loss = train_model(encoder, decoder, train_loader, criterion, optimizer, device)
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")

#     # Save Models
#     torch.save(encoder.state_dict(), "model_yash_encoder.pth")
#     torch.save(decoder.state_dict(), "model_yash_decoder.pth")
#     print("Models saved as 'model_yash_encoder.pth' and 'model_yash_decoder.pth'.")

#     # Evaluate BLEU score
#     encoder.eval()
#     decoder.eval()
#     references, hypotheses = [], []

#     with torch.no_grad():
#         for images, captions, lengths in tqdm(val_loader, desc="Evaluating"):
#             images = images.to(device)
#             features = encoder(images)
#             generated_caption = decoder.generate_caption(features, vocab)
#             references.append(captions[0].tolist())  # Use the first reference caption
#             hypotheses.append(generated_caption.split())

#     bleu_score = compute_bleu_score(references, hypotheses)
#     print(f"BLEU Score: {bleu_score:.4f}")

# if __name__ == "__main__":
#     main()


# import os
# import json
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, models
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# import random

# # Set random seed for reproducibility
# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)

# # --------------------
# # Vocabulary Class
# # --------------------
# class Vocabulary:
#     def __init__(self):
#         self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
#         self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
#         self.word_count = 4

#     def build_vocab(self, captions):
#         for caption in captions:
#             for word in caption.split():
#                 if word not in self.word2idx:
#                     self.word2idx[word] = self.word_count
#                     self.idx2word[self.word_count] = word
#                     self.word_count += 1

#     def __len__(self):
#         return len(self.word2idx)

#     def tokenize(self, caption):
#         return [self.word2idx.get(word, self.word2idx["<unk>"]) for word in caption.split()]

#     def detokenize(self, tokens):
#         return " ".join([self.idx2word[token] for token in tokens if token not in [0, 1, 2]])

# # --------------------
# # COCO Dataset Class
# # --------------------
# class CocoDataset(Dataset):
#     def __init__(self, img_dir, annotations_file, transform, vocab):
#         self.img_dir = img_dir
#         self.vocab = vocab
#         with open(annotations_file, "r") as f:
#             data = json.load(f)
#         self.annotations = data["annotations"]
#         self.image_info = {img["id"]: img["file_name"] for img in data["images"]}
#         self.transform = transform

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, idx):
#         annotation = self.annotations[idx]
#         img_path = os.path.join(self.img_dir, self.image_info[annotation["image_id"]])
#         caption = annotation["caption"]
#         image = Image.open(img_path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#         tokenized_caption = [self.vocab.word2idx["<start>"]] + \
#                             self.vocab.tokenize(caption) + \
#                             [self.vocab.word2idx["<end>"]]
#         return image, torch.tensor(tokenized_caption)

# # --------------------
# # Collate Function
# # --------------------
# def collate_fn(batch):
#     images, captions = zip(*batch)
#     images = torch.stack(images, dim=0)
#     caption_lengths = [len(cap) for cap in captions]
#     max_length = max(caption_lengths)
#     padded_captions = torch.zeros((len(captions), max_length), dtype=torch.long)
#     for i, cap in enumerate(captions):
#         padded_captions[i, :len(cap)] = cap
#     return images, padded_captions, torch.tensor(caption_lengths)

# # --------------------
# # Encoder
# # --------------------
# class EncoderCNN(nn.Module):
#     def __init__(self, embed_size):
#         super(EncoderCNN, self).__init__()
#         vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
#         self.features = vgg16.features
#         self.avgpool = vgg16.avgpool
#         self.fc = nn.Linear(25088, embed_size)  # 25088 is the output of VGG16 features

#     def forward(self, images):
#         features = self.features(images)
#         features = self.avgpool(features)
#         features = torch.flatten(features, 1)
#         embeddings = self.fc(features)
#         return embeddings

# # --------------------
# # Decoder
# # --------------------
# class DecoderRNN(nn.Module):
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
#         super(DecoderRNN, self).__init__()
#         self.embed = nn.Embedding(vocab_size, embed_size)
#         self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, vocab_size)

#     def forward(self, features, captions):
#         embeddings = self.embed(captions)
#         embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
#         hiddens, _ = self.lstm(embeddings)
#         outputs = self.fc(hiddens)
#         return outputs

# # --------------------
# # Training Function
# # --------------------
# def train_model(encoder, decoder, dataloader, criterion, optimizer, device):
#     encoder.train()
#     decoder.train()
#     total_loss = 0

#     for images, captions, lengths in tqdm(dataloader, desc="Training"):
#         images, captions = images.to(device), captions.to(device)
#         optimizer.zero_grad()

#         # Forward pass
#         features = encoder(images)
#         outputs = decoder(features, captions[:, :-1])  # Exclude <end> token from input

#         # Adjust output length to match target captions
#         target_length = captions[:, 1:].shape[1]  # Exclude <start> token from target
#         output_length = outputs.shape[1]

#         if output_length > target_length:
#             outputs = outputs[:, :target_length, :]  # Truncate
#         elif output_length < target_length:
#             padding = torch.zeros(
#                 (outputs.shape[0], target_length - output_length, outputs.shape[2]),
#                 device=outputs.device
#             )
#             outputs = torch.cat([outputs, padding], dim=1)  # Pad

#         # Compute loss
#         loss = criterion(outputs.reshape(-1, outputs.size(2)), captions[:, 1:].reshape(-1))  # Exclude <start> token
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     return total_loss / len(dataloader)

# # --------------------
# # Main Script
# # --------------------
# def main():
#     # Paths
#     train_images = "../COCO_Data/train2017"
#     train_captions = "../COCO_Data/annotations/captions_train2017.json"

#     # Hyperparameters
#     embed_size = 256
#     hidden_size = 512
#     batch_size = 64
#     num_epochs = 1
#     learning_rate = 1e-3
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Build Vocabulary
#     vocab = Vocabulary()
#     with open(train_captions, "r") as f:
#         annotations = json.load(f)["annotations"]
#     captions = [ann["caption"] for ann in annotations]
#     vocab.build_vocab(captions)

#     # Transforms
#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ])

#     # Datasets and Dataloaders
#     train_dataset = CocoDataset(train_images, train_captions, transform, vocab)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

#     # Models, Loss, and Optimizer
#     encoder = EncoderCNN(embed_size).to(device)
#     decoder = DecoderRNN(embed_size, hidden_size, len(vocab)).to(device)
#     criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<pad>"])
#     optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

#     # Training Loop
#     for epoch in range(num_epochs):
#         train_loss = train_model(encoder, decoder, train_loader, criterion, optimizer, device)
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")

# if __name__ == "__main__":
#     main()

######################################### RESNET50 ########################################################

# import os
# import json
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, models
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# import random

# # Set random seed for reproducibility
# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)

# # --------------------
# # Vocabulary Class
# # --------------------
# class Vocabulary:
#     def __init__(self):
#         self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
#         self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
#         self.word_count = 4

#     def build_vocab(self, captions):
#         for caption in captions:
#             for word in caption.split():
#                 if word not in self.word2idx:
#                     self.word2idx[word] = self.word_count
#                     self.idx2word[self.word_count] = word
#                     self.word_count += 1

#     def __len__(self):
#         return len(self.word2idx)

#     def tokenize(self, caption):
#         return [self.word2idx.get(word, self.word2idx["<unk>"]) for word in caption.split()]

#     def detokenize(self, tokens):
#         return " ".join([self.idx2word[token] for token in tokens if token not in [0, 1, 2]])

# # --------------------
# # COCO Dataset Class
# # --------------------
# class CocoDataset(Dataset):
#     def __init__(self, img_dir, annotations_file, transform, vocab):
#         self.img_dir = img_dir
#         self.vocab = vocab
#         with open(annotations_file, "r") as f:
#             data = json.load(f)
#         self.annotations = data["annotations"]
#         self.image_info = {img["id"]: img["file_name"] for img in data["images"]}
#         self.transform = transform

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, idx):
#         annotation = self.annotations[idx]
#         img_path = os.path.join(self.img_dir, self.image_info[annotation["image_id"]])
#         caption = annotation["caption"]
#         image = Image.open(img_path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#         tokenized_caption = [self.vocab.word2idx["<start>"]] + \
#                             self.vocab.tokenize(caption) + \
#                             [self.vocab.word2idx["<end>"]]
#         return image, torch.tensor(tokenized_caption)

# # --------------------
# # Collate Function
# # --------------------
# def collate_fn(batch):
#     images, captions = zip(*batch)
#     images = torch.stack(images, dim=0)
#     caption_lengths = [len(cap) for cap in captions]
#     max_length = max(caption_lengths)
#     padded_captions = torch.zeros((len(captions), max_length), dtype=torch.long)
#     for i, cap in enumerate(captions):
#         padded_captions[i, :len(cap)] = cap
#     return images, padded_captions, torch.tensor(caption_lengths)

# # --------------------
# # Encoder
# # --------------------
# class EncoderCNN(nn.Module):
#     def __init__(self, embed_size):
#         super(EncoderCNN, self).__init__()
#         resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
#         modules = list(resnet.children())[:-1]  # Remove the last FC layer
#         self.resnet = nn.Sequential(*modules)
#         self.embed = nn.Linear(resnet.fc.in_features, embed_size)

#     def forward(self, images):
#         features = self.resnet(images)
#         features = features.view(features.size(0), -1)
#         features = self.embed(features)
#         return features

# # --------------------
# # Decoder
# # --------------------
# class DecoderRNN(nn.Module):
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
#         super(DecoderRNN, self).__init__()
#         self.embed = nn.Embedding(vocab_size, embed_size)
#         self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, vocab_size)

#     def forward(self, features, captions):
#         embeddings = self.embed(captions)
#         embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
#         hiddens, _ = self.lstm(embeddings)
#         outputs = self.fc(hiddens)
#         return outputs

#     def generate_caption(self, features, vocab, max_len=20):
#         generated = []
#         inputs = features.unsqueeze(1)
#         states = None
#         for _ in range(max_len):
#             hiddens, states = self.lstm(inputs, states)
#             outputs = self.fc(hiddens.squeeze(1))
#             predicted = outputs.argmax(1)
#             word = vocab.idx2word[predicted.item()]
#             if word == "<end>":
#                 break
#             generated.append(word)
#             inputs = self.embed(predicted).unsqueeze(1)
#         return " ".join(generated)

# # --------------------
# # BLEU Score Calculation
# # --------------------
# def compute_bleu_score(references, hypotheses):
#     def ngram_counts(sequence, n):
#         return {tuple(sequence[i:i + n]): 1 for i in range(len(sequence) - n + 1)}

#     def precision(ref, hyp, n):
#         ref_ngrams = ngram_counts(ref, n)
#         hyp_ngrams = ngram_counts(hyp, n)
#         matched = sum(1 for ngram in hyp_ngrams if ngram in ref_ngrams)
#         return matched / max(1, len(hyp_ngrams))

#     precisions = [precision(ref, hyp, n) for n in range(1, 5)]
#     brevity_penalty = min(1.0, len(hyp) / len(ref))
#     bleu = brevity_penalty * np.exp(sum(np.log(p) if p > 0 else -1e6 for p in precisions) / 4)
#     return bleu

# # --------------------
# # Training Function
# # --------------------
# def train_model(encoder, decoder, dataloader, criterion, optimizer, device):
#     encoder.train()
#     decoder.train()
#     total_loss = 0

#     for images, captions, lengths in tqdm(dataloader, desc="Training"):
#         images, captions = images.to(device), captions.to(device)
#         optimizer.zero_grad()

#         features = encoder(images)
#         outputs = decoder(features, captions[:, :-1])  # Exclude <end> token from input

#         target_length = captions[:, 1:].shape[1]
#         output_length = outputs.shape[1]

#         if output_length > target_length:
#             outputs = outputs[:, :target_length, :]
#         elif output_length < target_length:
#             padding = torch.zeros((outputs.shape[0], target_length - output_length, outputs.shape[2]),
#                                    device=outputs.device)
#             outputs = torch.cat([outputs, padding], dim=1)

#         loss = criterion(outputs.reshape(-1, outputs.size(2)), captions[:, 1:].reshape(-1))  # Exclude <start> token
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     return total_loss / len(dataloader)

# # --------------------
# # Main Script
# # --------------------
# def main():
#     train_images = "../COCO_Data/train2017"
#     train_captions = "../COCO_Data/annotations/captions_train2017.json"
#     val_images = "../COCO_Data/val2017"
#     val_captions = "../COCO_Data/annotations/captions_val2017.json"

#     embed_size = 256
#     hidden_size = 512
#     batch_size = 128
#     num_epochs = 1
#     learning_rate = 1e-3
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     vocab = Vocabulary()
#     with open(train_captions, "r") as f:
#         annotations = json.load(f)["annotations"]
#     captions = [ann["caption"] for ann in annotations]
#     vocab.build_vocab(captions)

#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ])

#     train_dataset = CocoDataset(train_images, train_captions, transform, vocab)
#     val_dataset = CocoDataset(val_images, val_captions, transform, vocab)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
#     val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

#     encoder = EncoderCNN(embed_size).to(device)
#     decoder = DecoderRNN(embed_size, hidden_size, len(vocab)).to(device)
#     criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<pad>"])
#     optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

#     for epoch in range(num_epochs):
#         train_loss = train_model(encoder, decoder, train_loader, criterion, optimizer, device)
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")

#     torch.save(encoder.state_dict(), "model_yash_encoder.pth")
#     torch.save(decoder.state_dict(), "model_yash_decoder.pth")
#     print("Models saved as 'model_yash_encoder.pth' and 'model_yash_decoder.pth'.")

#     encoder.eval()
#     decoder.eval()
#     references, hypotheses = [], []

#     with torch.no_grad():
#         for images, captions, lengths in tqdm(val_loader, desc="Evaluating"):
#             images = images.to(device)
#             features = encoder(images)
#             generated_caption = decoder.generate_caption(features, vocab)
#             references.append(captions[0].tolist())
#             hypotheses.append([vocab.word2idx[word] for word in generated_caption.split()])

#     bleu_score = compute_bleu_score(references, hypotheses)
#     print(f"BLEU Score: {bleu_score:.4f}")

# if __name__ == "__main__":
#     main()



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


def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    caption_lengths = [len(cap) for cap in captions]
    max_length = max(caption_lengths)
    padded_captions = torch.zeros((len(captions), max_length), dtype=torch.long)
    for i, cap in enumerate(captions):
        padded_captions[i, :len(cap)] = cap
    return images, padded_captions


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1]  
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


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


# def train_model(encoder, decoder, dataloader, criterion, optimizer, device):
#     encoder.train()
#     decoder.train()
#     total_loss = 0
#     correct = 0
#     total = 0

#     for images, captions in tqdm(dataloader, desc="Training"):
#         images, captions = images.to(device), captions.to(device)
#         optimizer.zero_grad()

#         features = encoder(images)
#         outputs = decoder(features, captions[:, :-1])  # Exclude <end> token from input
#         loss = criterion(outputs.reshape(-1, outputs.size(2)), captions[:, 1:].reshape(-1))
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#         # Calculate accuracy
#         preds = outputs.argmax(dim=2)
#         correct += (preds[:, :-1] == captions[:, 1:]).sum().item()
#         total += captions[:, 1:].numel()

#     accuracy = 100 * correct / total
#     return total_loss / len(dataloader), accuracy

def train_model(encoder, decoder, dataloader, criterion, optimizer, device):
    encoder.train()
    decoder.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, captions in tqdm(dataloader, desc="Training"):
        images, captions = images.to(device), captions.to(device)
        optimizer.zero_grad()

        features = encoder(images)
        outputs = decoder(features, captions[:, :-1])  

       
        outputs = outputs[:, :captions.size(1) - 1, :]  

        loss = criterion(outputs.reshape(-1, outputs.size(2)), captions[:, 1:].reshape(-1))  
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        
        preds = outputs.argmax(dim=2)
        correct += (preds == captions[:, 1:]).sum().item()
        total += captions[:, 1:].numel()

    accuracy = 100 * correct / total
    return total_loss / len(dataloader), accuracy



def main():
    train_images = "../COCO_Data/train2017"
    train_captions = "../COCO_Data/annotations/captions_train2017.json"

    embed_size = 256
    hidden_size = 512
    batch_size = 256
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

<<<<<<< HEAD
    train_dataset = CocoDataset(train_images, train_captions, transform, vocab)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<pad>"])
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)


    with open("summary_yash.txt", "w") as f:
        f.write(str(encoder))
        f.write("\n\n")
        f.write(str(decoder))

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_model(encoder, decoder, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

    torch.save(encoder.state_dict(), "model_yash_resnet101_encoder.pth")
    torch.save(decoder.state_dict(), "model_yash_resnet101_decoder.pth")
    print("Models saved as 'model_yash_resnet101_encoder.pth' and 'model_yash_resnet101_decoder.pth'.")

if __name__ == "__main__":
    main()
=======
bleu = evaluate_model(encoder, decoder, val_loader)
<<<<<<< feature/yash
print(f"BLEU Score: {bleu:.4f}")
>>>>>>> origin/main

