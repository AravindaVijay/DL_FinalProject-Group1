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
