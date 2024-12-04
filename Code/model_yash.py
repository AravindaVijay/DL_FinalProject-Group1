import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Embedding, LSTM, Add, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

data, info = tfds.load('coco_captions', with_info=True)
train_data = data['train']
test_data = data['test']

captions = []
images = []

for example in train_data:
    image = example['image']
    caption_list = example['captions']
    for caption in caption_list:
        captions.append(caption)
        images.append(image.numpy())

print(f"Number of samples: {len(captions)}")
print("Sample Caption:", captions[0])
plt.imshow(images[0])
plt.axis('off')
plt.show()

tokenizer = Tokenizer(num_words=5000, oov_token="<unk>")
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1

tokenized_captions = tokenizer.texts_to_sequences(captions)
max_caption_length = 20
padded_captions = pad_sequences(tokenized_captions, maxlen=max_caption_length, padding='post')

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

def build_rnn(vocab_size, max_caption_length):
    inputs = Input(shape=(max_caption_length,))
    x = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(inputs)
    x = LSTM(256, return_sequences=False)(x)
    x = Dense(256, activation='relu')(x)
    return tf.keras.Model(inputs, x)

rnn = build_rnn(vocab_size=vocab_size, max_caption_length=max_caption_length)
rnn.summary()

def build_image_captioning_model(vocab_size, max_caption_length):
    image_input = cnn.input
    image_features = cnn.output
    caption_input = rnn.input
    caption_features = rnn.output
    combined = Add()([image_features, caption_features])
    output = Dense(vocab_size, activation='softmax')(combined)
    return Model(inputs=[image_input, caption_input], outputs=output)

model = build_image_captioning_model(vocab_size, max_caption_length)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.summary()

def preprocess_image(image):
    image = tf.image.resize(image, (224, 224)) / 255.0
    return image

def create_sequences(tokenized_captions, images, max_caption_length):
    X1, X2, y = [], [], []
    for i, caption in enumerate(tokenized_captions):
        for t in range(1, len(caption)):
            in_seq = caption[:t]
            out_seq = caption[t]
            in_seq = pad_sequences([in_seq], maxlen=max_caption_length, padding='post')[0]
            image = preprocess_image(images[i])
            features = cnn(tf.expand_dims(image, axis=0))
            X1.append(features.numpy().flatten())
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

num_samples = 1000
X1, X2, y = create_sequences(tokenized_captions[:num_samples], images[:num_samples], max_caption_length)
y = to_categorical(y, num_classes=vocab_size)

model.fit([X1, X2], y, batch_size=32, epochs=10)

def generate_caption(model, tokenizer, image, max_caption_length):
    image = preprocess_image(image)
    features = cnn(tf.expand_dims(image, axis=0)).numpy().flatten().reshape(1, -1)
    input_seq = [tokenizer.word_index['<start>']]
    for _ in range(max_caption_length):
        sequence = pad_sequences([input_seq], maxlen=max_caption_length, padding='post')
        yhat = model.predict([features, sequence], verbose=0)
        word_index = np.argmax(yhat)
        word = tokenizer.index_word.get(word_index, '<unk>')
        if word == '<end>':
            break
        input_seq.append(word_index)
    return ' '.join([tokenizer.index_word[i] for i in input_seq[1:]])

test_image, _ = next(iter(test_data))
plt.imshow(test_image.numpy())
plt.axis('off')
plt.show()

caption = generate_caption(model, tokenizer, test_image, max_caption_length)
print("Generated Caption:", caption)
