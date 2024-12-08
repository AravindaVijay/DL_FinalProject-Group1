from torchvision import transforms
from model import ImageEncoder, CaptionDecoder
import torch
import pickle
import os

transform_test = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),  # normalize image for pre-trained model
            (0.229, 0.224, 0.225),
        ),
    ]
)

vocab_file = 'vocab.pkl'

try:
    with open(vocab_file, "rb") as f:
        vocab = pickle.load(f)
    print("Vocabulary successfully loaded from vocab.pkl file!")
except FileNotFoundError:
    print("Vocabulary file not found!")
except Exception as e:
    print(f"An error occurred while loading the vocabulary: {e}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embedding_dim = 256
hidden_dim = 512
vocab_size = len(vocab)


encoder = ImageEncoder(embedding_dim).to(device)
decoder = CaptionDecoder(embedding_dim, hidden_dim, vocab_size, vocab).to(device)

encoder.load_state_dict(torch.load('encoder.pth'))
decoder.load_state_dict(torch.load('decoder.pth'))


def generate_caption(image, max_length=200):
    image = transform_test(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = encoder(image)
    generated_list = decoder.generate_caption(image_features, max_length=max_length)
    generated_caption = " ".join(generated_list)
    generated_caption = generated_caption.replace("<start>", "").replace("<end>", "")
    return generated_caption