<<<<<<< HEAD
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Embedding, LSTM, Add, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Load and Preprocess Dataset
# -------------------------------

# Load Flickr8k dataset
data, info = tfds.load('flickr8k', with_info=True, as_supervised=True)
train_data = data['train']
test_data = data['test']

# Collect captions and images
captions = []
images = []

for image, caption in train_data:
    captions.append(caption.numpy().decode('utf-8'))
    images.append(image)

# Tokenize captions
tokenizer = Tokenizer(num_words=5000, oov_token="<unk>")
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1  # Include padding token
print(f"Vocabulary Size: {vocab_size}")

# Convert captions to sequences
tokenized_captions = tokenizer.texts_to_sequences(captions)

# Pad captions
max_caption_length = 20
padded_captions = pad_sequences(tokenized_captions, maxlen=max_caption_length, padding='post')

# -------------------------------
# Step 2: Define Custom CNN
# -------------------------------

def build_cnn():
    inputs = Input(shape=(224, 224, 3))
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    return tf.keras.Model(inputs, x)

cnn = build_cnn()
cnn.summary()

# -------------------------------
# Step 3: Define Custom RNN
# -------------------------------

def build_rnn(vocab_size, max_caption_length):
    inputs = Input(shape=(max_caption_length,))
    x = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(inputs)
    x = LSTM(256, return_sequences=False)(x)
    x = Dense(256, activation='relu')(x)
    return tf.keras.Model(inputs, x)

rnn = build_rnn(vocab_size=vocab_size, max_caption_length=max_caption_length)
rnn.summary()

# -------------------------------
# Step 4: Combine CNN and RNN
# -------------------------------

def build_image_captioning_model(vocab_size, max_caption_length):
    # Image input and features
    image_input = cnn.input
    image_features = cnn.output

    # Caption input and features
    caption_input = rnn.input
    caption_features = rnn.output

    # Combine image and caption features
    combined = Add()([image_features, caption_features])
    output = Dense(vocab_size, activation='softmax')(combined)

    # Define model
    return Model(inputs=[image_input, caption_input], outputs=output)

model = build_image_captioning_model(vocab_size, max_caption_length)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.summary()

# -------------------------------
# Step 5: Prepare Data for Training
# -------------------------------

def preprocess_image(image):
    image = tf.image.resize(image, (224, 224)) / 255.0  # Normalize
    return image

def create_sequences(tokenized_captions, images, max_caption_length):
    X1, X2, y = [], [], []
    for i, caption in enumerate(tokenized_captions):
        for t in range(1, len(caption)):
            # Input sequence
            in_seq = caption[:t]
            # Output word
            out_seq = caption[t]
            # Pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_caption_length, padding='post')[0]
            # Preprocess image
            image = preprocess_image(images[i])
            features = cnn(tf.expand_dims(image, axis=0))
            # Append data
            X1.append(features.numpy().flatten())
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

# Prepare training data
X1, X2, y = create_sequences(tokenized_captions, images, max_caption_length)
y = to_categorical(y, num_classes=vocab_size)
print(f"Training Data Shapes: X1={X1.shape}, X2={X2.shape}, y={y.shape}")

# -------------------------------
# Step 6: Train the Model
# -------------------------------

model.fit([X1, X2], y, batch_size=32, epochs=10)

# -------------------------------
# Step 7: Generate Captions
# -------------------------------

def generate_caption(model, tokenizer, image, max_caption_length):
    # Preprocess the image
    image = preprocess_image(image)
    features = cnn(tf.expand_dims(image, axis=0)).numpy().flatten().reshape(1, -1)
    # Start generating caption
    input_seq = [tokenizer.word_index['<start>']]
    for _ in range(max_caption_length):
        # Pad sequence
        sequence = pad_sequences([input_seq], maxlen=max_caption_length, padding='post')
        # Predict next word
        yhat = model.predict([features, sequence], verbose=0)
        word_index = np.argmax(yhat)
        word = tokenizer.index_word.get(word_index, '<unk>')
        if word == '<end>':
            break
        input_seq.append(word_index)
    return ' '.join([tokenizer.index_word[i] for i in input_seq[1:]])

# -------------------------------
# Step 8: Test the Model
# -------------------------------

test_image, _ = next(iter(test_data))  # Load a test image
plt.imshow(test_image.numpy())
plt.axis('off')
plt.show()

# Generate caption
caption = generate_caption(model, tokenizer, test_image, max_caption_length)
print("Generated Caption:", caption)
=======
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
# caption = generate_caption(model, tokenizer, test_image, max_caption_length)
# print("Generated Caption:", caption)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm
from pycocotools.coco import COCO
from nltk.translate.bleu_score import corpus_bleu
from data_loader import get_data_loader
from vocabulary import Vocabulary
from torch.cuda.amp import GradScaler, autocast
import numpy as np

# Set random seed for reproducibility
import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Paths
annotations_folder = "../COCO_Data/annotations"
train_images_folder = "../COCO_Data/train2017"
val_images_folder = "../COCO_Data/val2017"
captions_file = os.path.join(annotations_folder, "captions_train2017.json")
val_captions_file = os.path.join(annotations_folder, "captions_val2017.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vocabulary
vocab = Vocabulary(annotations_file=captions_file, vocab_exists=True)

# Image Augmentation
train_transform = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Attention Mechanism
class Attention(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(feature_dim + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, image_features, hidden_state):
        hidden_state = hidden_state.unsqueeze(1).repeat(1, image_features.size(1), 1)
        combined_features = torch.cat((image_features, hidden_state), dim=2)
        attention_weights = torch.softmax(self.v(torch.tanh(self.attention(combined_features))), dim=1)
        attended_features = (attention_weights * image_features).sum(dim=1)
        return attended_features, attention_weights

# Encoder using ResNet-101
class ResNet101Encoder(nn.Module):
    def __init__(self, embedding_dim):
        super(ResNet101Encoder, self).__init__()
        resnet = models.resnet101(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # Use all layers except the classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, embedding_dim)

        # Freeze the ResNet layers
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, images):
        features = self.features(images)  # Extract feature maps
        features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)  # Flatten
        return self.fc(features)

# Load Pre-trained Embeddings (e.g., GloVe)
def load_pretrained_embeddings(vocab, embedding_dim, embedding_path):
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    with open(embedding_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            if word in vocab.word2idx:
                idx = vocab.word2idx[word]
                embedding_matrix[idx] = vector
    return torch.tensor(embedding_matrix, dtype=torch.float32)

# Decoder with Pre-trained Embeddings
class CaptionDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, vocab, embedding_weights=None, num_layers=1):
        super(CaptionDecoder, self).__init__()
        if embedding_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.attention = Attention(embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, image_features, captions):
        embeddings = self.embedding(captions[:, :-1])
        h, c = torch.zeros((1, image_features.size(0), hidden_dim)).to(device), torch.zeros((1, image_features.size(0), hidden_dim)).to(device)
        outputs = []
        for t in range(embeddings.size(1)):
            attended_features, _ = self.attention(image_features, h[-1])
            lstm_input = torch.cat((attended_features.unsqueeze(1), embeddings[:, t].unsqueeze(1)), dim=2)
            output, (h, c) = self.lstm(lstm_input, (h, c))
            outputs.append(self.fc(output.squeeze(1)))
        return torch.stack(outputs, dim=1)

# Training Setup
embedding_dim = 300  # Match pre-trained embedding dimensions
hidden_dim = 512
vocab_size = len(vocab)

# Load pre-trained GloVe embeddings
pretrained_embedding_path = "../glove.6B.300d.txt"
embedding_weights = load_pretrained_embeddings(vocab, embedding_dim, pretrained_embedding_path)

# Initialize models
encoder = ResNet101Encoder(embedding_dim).to(device)
decoder = CaptionDecoder(embedding_dim, hidden_dim, vocab_size, vocab, embedding_weights).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<pad>'])
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

train_loader = get_data_loader(train_images_folder, captions_file, vocab_exists=True, batch_size=64, transform=train_transform)
val_loader = get_data_loader(val_images_folder, val_captions_file, vocab_exists=True, batch_size=1, transform=val_transform)

# Mixed Precision Training
scaler = GradScaler()

# Training Loop
num_epochs = 20
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = batch['images'].to(device)
        captions = batch['tokenized_caption'].to(device)

        optimizer.zero_grad()

        with autocast():
            image_features = encoder(images)
            outputs = decoder(image_features, captions)
            loss = criterion(outputs.view(-1, vocab_size), captions[:, 1:].reshape(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    scheduler.step()

torch.save(encoder.state_dict(), "encoder_resnet101_glove.pth")
torch.save(decoder.state_dict(), "decoder_resnet101_glove.pth")

# Evaluation
def evaluate_model(encoder, decoder, val_loader, max_length=20):
    encoder.eval()
    decoder.eval()
    references, hypotheses = [], []

    for batch in tqdm(val_loader, desc="Evaluating"):
        images = batch['images'].to(device)
        all_captions = batch['all_captions']

        with torch.no_grad():
            image_features = encoder(images)
            generated_caption = decoder.generate_caption(image_features, max_length=max_length)

        hypotheses.append(generated_caption.split())
        references.append([caption.split() for caption in all_captions])

    bleu_score = corpus_bleu(references, hypotheses)
    return bleu_score

encoder.load_state_dict(torch.load("encoder_resnet101_glove.pth"))
decoder.load_state_dict(torch.load("decoder_resnet101_glove.pth"))

bleu = evaluate_model(encoder, decoder, val_loader)
print(f"BLEU Score: {bleu:.4f}")
>>>>>>> origin/main
