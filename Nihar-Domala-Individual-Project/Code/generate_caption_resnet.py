from torchvision import transforms
from model_resnet import EncoderCNN, DecoderRNN
import torch
import pickle
import os
from PIL import Image

cocoapi_dir = r"../COCO_Data"

# Defining a transform to pre-process the testing images.
transform_test = transforms.Compose(
    [
        transforms.Resize(256),  # smaller edge of image resized to 256
        transforms.RandomCrop(224),  # get 224x224 crop from random location
        transforms.RandomHorizontalFlip(),  # horizontally flip image with probability=0.5
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize(
            (0.485, 0.456, 0.406),  # normalize image for pre-trained model
            (0.229, 0.224, 0.225),
        ),
    ]
)

vocab_file = 'vocab_2.pkl'

try:
    with open(vocab_file, "rb") as f:
        vocab = pickle.load(f)
    print("Vocabulary successfully loaded from vocab.pkl file!")
except FileNotFoundError:
    print("Vocabulary file not found!")
except Exception as e:
    print(f"An error occurred while loading the vocabulary: {e}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Select appropriate values for the Python variables below.
embed_size = 256
hidden_size = 512

# The size of the vocabulary.
vocab_size = len(vocab)

# Initialize the encoder and decoder, and set each to inference mode.
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
encoder.eval()
decoder.eval()

# Load the trained weights.
encoder.load_state_dict(torch.load('encoder_resnet.pkl'))
decoder.load_state_dict(torch.load('decoder_resnet.pkl'))

# Move models to GPU if CUDA is available.
encoder.to(device)
decoder.to(device)


def clean_sentence(output, idx2word):
    sentence = ""
    for i in output:
        word = idx2word[i]
        if i == 0:
            continue
        if i == 1:
            break
        else:
            sentence = sentence + " " + word
    return sentence


def generate_caption_resnet(image, max_length=200):
    image = transform_test(image).unsqueeze(0).to(device)
    features = encoder(image).unsqueeze(1)
    output = decoder.sample(features)
    idx2word = vocab.idx2word
    sentence = clean_sentence(output, idx2word)
    return sentence


# try the function
if __name__ == "__main__":
    image = Image.open("../../COCO_Data/val2017/000000000139.jpg").convert("RGB")
    caption = generate_caption_resnet(image)
    print(caption)