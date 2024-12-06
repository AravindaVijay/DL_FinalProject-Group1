import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Embedding, LSTM, Add, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

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


# import tensorflow as tf
# from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Embedding, LSTM, Concatenate
# from tensorflow.keras.models import Model
# import json
# import os
# import pickle
# import numpy as np
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import to_categorical
# import cv2


# annotations_folder = r"/home/ubuntu/DL_FinalProject-Group1/coco_dataset/annotations"
# train_images_folder = r"/home/ubuntu/DL_FinalProject-Group1/coco_dataset/train2017"
# val_images_folder = r"/home/ubuntu/DL_FinalProject-Group1/coco_dataset/val2017"


# train_annotations_path = os.path.join(annotations_folder, "captions_train2017.json")
# val_annotations_path = os.path.join(annotations_folder, "captions_val2017.json")


# def process_annotations(annotations_path, images_folder):
    
#     with open(annotations_path, 'r') as f:
#         annotations = json.load(f)

    
#     image_caption_map = {}
#     for ann in annotations['annotations']:
#         image_id = ann['image_id'] 
#         caption = ann['caption']  
#         if image_id not in image_caption_map:
#             image_caption_map[image_id] = []
#         image_caption_map[image_id].append(caption)

    
#     image_id_to_path = {}
#     for image_info in annotations['images']:
#         image_id = image_info['id']  
#         filename = image_info['file_name']  
#         image_id_to_path[image_id] = os.path.join(images_folder, filename)


#     image_path_to_captions = {}
#     for image_id, captions in image_caption_map.items():
#         if image_id in image_id_to_path:  
#             image_path = image_id_to_path[image_id]
#             image_path_to_captions[image_path] = captions

#     return image_path_to_captions

# print("Processing train2017 annotations...")
# train_image_path_to_captions = process_annotations(train_annotations_path, train_images_folder)

# print("Processing val2017 annotations...")
# val_image_path_to_captions = process_annotations(val_annotations_path, val_images_folder)

# with open("train_image_path_to_captions.pkl", "wb") as f:
#     pickle.dump(train_image_path_to_captions, f)

# with open("val_image_path_to_captions.pkl", "wb") as f:
#     pickle.dump(val_image_path_to_captions, f)


# print("Sample mappings from train2017:")
# for image_path, captions in list(train_image_path_to_captions.items())[:5]:
#     print(f"Image Path: {image_path}")
#     print(f"Captions: {captions}")
#     print("-" * 80)

# print("Sample mappings from val2017:")
# for image_path, captions in list(val_image_path_to_captions.items())[:5]:
#     print(f"Image Path: {image_path}")
#     print(f"Captions: {captions}")
#     print("-" * 80)

# with open("tokenizer.pkl", "rb") as f:
#     tokenizer = pickle.load(f)


# def build_captioning_model(vocab_size, max_length):
#     # Image feature extraction
#     image_input = Input(shape=(128, 128, 3), name="Image_Input")  # Image input (128x128 RGB)
#     x = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
#     x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#     x = Flatten()(x)
#     image_features = Dense(256, activation='relu', name="Image_Features")(x)

#     # Caption input processing
#     caption_input = Input(shape=(max_length,), name="Caption_Input")  # Caption input (sequence)
#     embedding = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(caption_input)
#     lstm = LSTM(256)(embedding)

#     # Combine features
#     combined = Concatenate()([image_features, lstm])
#     decoder = Dense(512, activation='relu')(combined)
#     output = Dense(vocab_size, activation='softmax')(decoder)

   
#     model = Model(inputs=[image_input, caption_input], outputs=output)
#     model.compile(optimizer='adam', loss='categorical_crossentropy')
#     return model


# vocab_size = len(tokenizer.word_index) + 1  
# all_captions = []
# for captions in train_image_path_to_captions.values():
#     all_captions.extend(captions)

# caption_sequences = tokenizer.texts_to_sequences(all_captions)
# max_length = max(len(seq) for seq in caption_sequences)

# model = build_captioning_model(vocab_size, max_length)
# model.summary()

# # Preprocess images
# def preprocess_image(image_path):
#     """Resize and normalize image."""
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (128, 128))  # Resize to model input size
#     img = img / 255.0  # Normalize pixel values to [0, 1]
#     return img

# # Data generator
# def data_generator(image_path_to_captions, tokenizer, batch_size, max_length, vocab_size):
#     while True:
#         batch_images = []
#         batch_captions = []
#         batch_labels = []
        
#         for image_path, captions in list(image_path_to_captions.items()):
#             img = preprocess_image(image_path)  # Preprocess image

#             for caption in captions:
#                 seq = tokenizer.texts_to_sequences([caption])[0]
#                 for i in range(1, len(seq)):
#                     input_seq, output_word = seq[:i], seq[i]
#                     input_seq = pad_sequences([input_seq], maxlen=max_length, padding='post')[0]
#                     output_word = to_categorical([output_word], num_classes=vocab_size)[0]
                    
#                     batch_images.append(img)
#                     batch_captions.append(input_seq)
#                     batch_labels.append(output_word)

#                     if len(batch_images) == batch_size:
#                         yield [np.array(batch_images), np.array(batch_captions)], np.array(batch_labels)
#                         batch_images, batch_captions, batch_labels = [], [], []

# # Load train and validation mappings
# with open("train_image_path_to_captions.pkl", "rb") as f:
#     train_image_path_to_captions = pickle.load(f)

# with open("val_image_path_to_captions.pkl", "rb") as f:
#     val_image_path_to_captions = pickle.load(f)

# # Load tokenizer
# with open("tokenizer.pkl", "rb") as f:
#     tokenizer = pickle.load(f)

# vocab_size = 5000  
# max_length = 20  

# batch_size = 32

# # Create training generator
# train_gen = data_generator(train_image_path_to_captions, tokenizer, batch_size, max_length, vocab_size)

# # Train the model
# model.fit(
#     train_gen,
#     steps_per_epoch=len(train_image_path_to_captions) // batch_size,
#     epochs=10
# )

# def generate_caption(image_path, tokenizer, max_length):
#     """Generate a caption for a given image."""
#     img = preprocess_image(image_path)
#     img = np.expand_dims(img, axis=0)  # Add batch dimension

#     input_seq = [tokenizer.word_index['<start>']]  # Start token
#     for _ in range(max_length):
#         padded_seq = pad_sequences([input_seq], maxlen=max_length, padding='post')
#         pred = model.predict([img, padded_seq], verbose=0)
#         next_word = tokenizer.index_word.get(np.argmax(pred), '<unk>')
#         input_seq.append(np.argmax(pred))
#         if next_word == '<end>':
#             break
#     return ' '.join([tokenizer.index_word[idx] for idx in input_seq[1:-1]])

# # Example usage
# example_image_path = "/path/to/image.jpg"
# caption = generate_caption(example_image_path, tokenizer, max_length)
# print("Generated Caption:", caption)

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Embedding, LSTM, Concatenate
from tensorflow.keras.models import Model
import json
import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import cv2

annotations_folder = r"/home/ubuntu/DL_FinalProject-Group1/coco_dataset/annotations"
train_images_folder = r"/home/ubuntu/DL_FinalProject-Group1/coco_dataset/train2017"
val_images_folder = r"/home/ubuntu/DL_FinalProject-Group1/coco_dataset/val2017"

train_annotations_path = os.path.join(annotations_folder, "captions_train2017.json")
val_annotations_path = os.path.join(annotations_folder, "captions_val2017.json")

def process_annotations(annotations_path, images_folder):
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    image_caption_map = {}
    for ann in annotations['annotations']:
        image_id = ann['image_id']
        caption = ann['caption']
        if image_id not in image_caption_map:
            image_caption_map[image_id] = []
        image_caption_map[image_id].append(caption)
    image_id_to_path = {}
    for image_info in annotations['images']:
        image_id = image_info['id']
        filename = image_info['file_name']
        image_id_to_path[image_id] = os.path.join(images_folder, filename)
    image_path_to_captions = {}
    for image_id, captions in image_caption_map.items():
        if image_id in image_id_to_path:
            image_path = image_id_to_path[image_id]
            image_path_to_captions[image_path] = captions
    return image_path_to_captions

train_image_path_to_captions = process_annotations(train_annotations_path, train_images_folder)
val_image_path_to_captions = process_annotations(val_annotations_path, val_images_folder)

with open("train_image_path_to_captions.pkl", "wb") as f:
    pickle.dump(train_image_path_to_captions, f)

with open("val_image_path_to_captions.pkl", "wb") as f:
    pickle.dump(val_image_path_to_captions, f)

if os.path.exists("tokenizer.pkl"):
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
else:
    all_captions = []
    for captions in train_image_path_to_captions.values():
        all_captions.extend(captions)
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000, oov_token="<unk>")
    tokenizer.fit_on_texts(all_captions)
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

vocab_size = len(tokenizer.word_index) + 1
all_captions = []
for captions in train_image_path_to_captions.values():
    all_captions.extend(captions)
caption_sequences = tokenizer.texts_to_sequences(all_captions)
max_length = max(len(seq) for seq in caption_sequences)

def build_captioning_model(vocab_size, max_length):
    image_input = Input(shape=(128, 128, 3), name="Image_Input")
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)
    image_features = Dense(256, activation='relu', name="Image_Features")(x)
    caption_input = Input(shape=(max_length,), name="Caption_Input")
    embedding = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(caption_input)
    lstm = LSTM(256)(embedding)
    combined = Concatenate()([image_features, lstm])
    decoder = Dense(512, activation='relu')(combined)
    output = Dense(vocab_size, activation='softmax')(decoder)
    model = Model(inputs=[image_input, caption_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

model = build_captioning_model(vocab_size, max_length)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    return img

def data_generator(image_path_to_captions, tokenizer, batch_size, max_length, vocab_size):
    while True:
        batch_images = []
        batch_captions = []
        batch_labels = []
        for image_path, captions in list(image_path_to_captions.items()):
            assert os.path.exists(image_path), f"Image path not found: {image_path}"
            img = preprocess_image(image_path)
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    input_seq, output_word = seq[:i], seq[i]
                    input_seq = pad_sequences([input_seq], maxlen=max_length, padding='post')[0]
                    output_word = to_categorical([output_word], num_classes=vocab_size)[0]
                    batch_images.append(img)
                    batch_captions.append(input_seq)
                    batch_labels.append(output_word)
                    if len(batch_images) == batch_size:
                        yield (np.array(batch_images), np.array(batch_captions)), np.array(batch_labels)
                        batch_images, batch_captions, batch_labels = [], [], []

batch_size = 32
train_gen = data_generator(train_image_path_to_captions, tokenizer, batch_size, max_length, vocab_size)

model.fit(
    train_gen,
    steps_per_epoch=len(train_image_path_to_captions) // batch_size,
    epochs=10
)

def generate_caption(image_path, tokenizer, max_length):
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)
    input_seq = [tokenizer.word_index['<start>']]
    for _ in range(max_length):
        padded_seq = pad_sequences([input_seq], maxlen=max_length, padding='post')
        pred = model.predict([img, padded_seq], verbose=0)
        next_word = tokenizer.index_word.get(np.argmax(pred), '<unk>')
        input_seq.append(np.argmax(pred))
        if next_word == '<end>':
            break
    return ' '.join([tokenizer.index_word[idx] for idx in input_seq[1:-1]])

example_image_path = "/path/to/image.jpg"
caption = generate_caption(example_image_path, tokenizer, max_length)
print("Generated Caption:", caption)


####### Validating Captins and images

import json
import os
import pickle
import random
from PIL import Image
import matplotlib.pyplot as plt

# Paths to the COCO dataset
annotations_folder = "coco_dataset/annotations"
train_images_folder = "coco_dataset/train2017"
val_images_folder = "coco_dataset/val2017"

# Annotations files
train_annotations_path = os.path.join(annotations_folder, "captions_train2017.json")
val_annotations_path = os.path.join(annotations_folder, "captions_val2017.json")

# Function to process annotations
def process_annotations(annotations_path, images_folder):
    # Load Annotations JSON
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    # Map Image IDs to Captions
    image_caption_map = {}
    for ann in annotations['annotations']:
        image_id = ann['image_id']
        caption = ann['caption']
        if image_id not in image_caption_map:
            image_caption_map[image_id] = []
        image_caption_map[image_id].append(caption)

    # Map Image IDs to File Paths
    image_id_to_path = {}
    for image_info in annotations['images']:
        image_id = image_info['id']
        filename = image_info['file_name']
        image_id_to_path[image_id] = os.path.join(images_folder, filename)

    # Combine Image Paths and Captions
    image_path_to_captions = {}
    for image_id, captions in image_caption_map.items():
        if image_id in image_id_to_path:
            image_path = image_id_to_path[image_id]
            image_path_to_captions[image_path] = captions

    return image_path_to_captions

# Process train2017 and val2017 annotations
print("Processing train2017 annotations...")
train_image_path_to_captions = process_annotations(train_annotations_path, train_images_folder)

print("Processing val2017 annotations...")
val_image_path_to_captions = process_annotations(val_annotations_path, val_images_folder)

# Save the mappings for later use
with open("train_image_path_to_captions.pkl", "wb") as f:
    pickle.dump(train_image_path_to_captions, f)

with open("val_image_path_to_captions.pkl", "wb") as f:
    pickle.dump(val_image_path_to_captions, f)

print("Mappings saved successfully!")

# Validation Functions
def check_total_images(folder_path, mapping):
    total_images = len(os.listdir(folder_path))
    mapped_images = len(mapping)
    print(f"Total images in {folder_path}: {total_images}")
    print(f"Mapped images: {mapped_images}")
    unmapped_files = total_images - mapped_images
    if unmapped_files > 0:
        print(f"Unmapped images: {unmapped_files}")
    else:
        print("All images are mapped successfully!")

def display_sample_images(mapping, num_samples=3):
    sample_paths = random.sample(list(mapping.keys()), num_samples)
    for path in sample_paths:
        captions = mapping[path]
        print(f"Image Path: {path}")
        print(f"Captions: {captions}")
        image = Image.open(path)
        plt.imshow(image)
        plt.axis("off")
        plt.title("\n".join(captions), fontsize=12)
        plt.show()

def validate_image_paths(mapping):
    invalid_paths = [path for path in mapping.keys() if not os.path.exists(path)]
    if invalid_paths:
        print(f"Invalid image paths: {len(invalid_paths)}")
        print(f"Examples: {invalid_paths[:5]}")
    else:
        print("All image paths are valid!")

# Perform validation
print("\n--- Validation Results ---")
print("\nChecking train2017 mappings:")
check_total_images(train_images_folder, train_image_path_to_captions)
validate_image_paths(train_image_path_to_captions)
print("\nDisplaying random samples from train2017:")
display_sample_images(train_image_path_to_captions)

print("\nChecking val2017 mappings:")
check_total_images(val_images_folder, val_image_path_to_captions)
validate_image_paths(val_image_path_to_captions)
print("\nDisplaying random samples from val2017:")
display_sample_images(val_image_path_to_captions)
