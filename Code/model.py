##########################################################################
######################PRETRAINED RESNET AND LSTM########################
##########################################################################

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import random
import numpy as np
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

###################################################################################################################
# model 1 - ImageEncoder and CaptionDecoder, need to check training
###################################################################################################################
class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim, trainable_layers=2):
        super(ImageEncoder, self).__init__()
        base_model = models.resnet50(pretrained=True)
        for param in base_model.parameters():
            param.requires_grad = False
        for param in list(base_model.parameters())[-trainable_layers:]:
            param.requires_grad = True
        feature_extractor = list(base_model.children())[:-1]
        self.feature_extractor = nn.Sequential(*feature_extractor)
        self.fc = nn.Linear(base_model.fc.in_features, embedding_dim)

    def forward(self, images):
        extracted_features = self.feature_extractor(images)
        extracted_features = extracted_features.view(extracted_features.size(0), -1)
        embeddings = self.fc(extracted_features)
        return embeddings

class CaptionDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, vocab, num_layers=1):
        super(CaptionDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_state = (torch.zeros(1, 1, hidden_dim), torch.zeros(1, 1, hidden_dim))
        self.end_token_id = vocab(vocab.end_word)  # Add the end token ID

    def forward(self, image_features, captions):
        caption_embeddings = self.word_embeddings(captions[:, :-1])
        inputs = torch.cat((image_features.unsqueeze(1), caption_embeddings), dim=1)
        lstm_output, self.hidden_state = self.lstm(inputs)
        outputs = self.fc(lstm_output)
        return outputs

    def generate_caption(self, inputs, states=None, max_length=20):
        """
        Generate a caption for a given image feature vector.
        Args:
            inputs: Image feature vector (shape: [batch_size, embedding_dim]).
            states: Initial hidden and cell states for LSTM.
            max_length: Maximum length of the generated caption.
        Returns:
            generated_ids: List of word indices forming the generated caption.
        """
        batch_size = inputs.size(0)  # Get the batch size
        if states is None:
            # Initialize hidden and cell states as 3D tensors
            states = (
                torch.zeros(1, batch_size, self.hidden_dim).to(inputs.device),  # hx
                torch.zeros(1, batch_size, self.hidden_dim).to(inputs.device)   # cx
            )

        generated_ids = []
        inputs = inputs.unsqueeze(1)  # Add sequence dimension for LSTM input

        for _ in range(max_length):
            lstm_output, states = self.lstm(inputs, states)
            outputs = self.fc(lstm_output.squeeze(1))  # Predict next word
            predicted_token = outputs.argmax(dim=1)  # Get the token with max probability
            generated_ids.append(predicted_token.item())

            # If <end> token is generated, stop
            if predicted_token.item() == self.end_token_id:
                break

            # Prepare input for the next timestep
            inputs = self.word_embeddings(predicted_token).unsqueeze(1)

        return generated_ids


# # import torch.nn as nn
# # import torchvision.models as models

# # class ImageEncoder(nn.Module):
# #     def __init__(self, embedding_dim, trainable_layers=2):
# #         super(ImageEncoder, self).__init__()
# #         base_model = models.resnet50(pretrained=True)
# #         for param in base_model.parameters():
# #             param.requires_grad = False
# #         for param in list(base_model.parameters())[-trainable_layers:]:
# #             param.requires_grad = True
# #         self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
# #         self.fc = nn.Linear(base_model.fc.in_features, embedding_dim)

# #     def forward(self, images):
# #         features = self.feature_extractor(images)
# #         features = features.view(features.size(0), -1)  # Flatten
# #         embeddings = self.fc(features)
# #         return embeddings

# # class CaptionDecoder(nn.Module):
# #     def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers=1):
# #         super(CaptionDecoder, self).__init__()
# #         self.embedding = nn.Embedding(vocab_size, embedding_dim)
# #         self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
# #         self.fc = nn.Linear(hidden_dim, vocab_size)

# #     def forward(self, features, captions):
# #         embeddings = self.embedding(captions[:, :-1])  # Exclude <end> token
# #         inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
# #         lstm_output, _ = self.lstm(inputs)
# #         outputs = self.fc(lstm_output)
# #         return outputs

##########################################################################
###################### CNN AND RNN ######################## Currently used in training loop
##########################################################################
#%%
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import transforms
from data_loader import get_data_loader
import torch.optim as optim
from vocabulary import Vocabulary
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu
from pycocotools.coco import COCO
import nltk

vocab = Vocabulary(annotations_file='../COCO_Data/annotations/captions_train2017.json', vocab_exists=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomCNNEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(CustomCNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)  # Downscale image
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Pool to 1x1 feature map
        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, images):
        x = torch.relu(self.conv1(images))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class CustomRNNDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers=1):
        super(CustomRNNDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, image_features, captions):
        embeddings = self.embedding(captions[:, :-1])  # Exclude <end> token
        inputs = torch.cat((image_features.unsqueeze(1), embeddings), dim=1)
        rnn_output, _ = self.rnn(inputs)
        outputs = self.fc(rnn_output)
        return outputs

    def generate_caption(self, inputs, states=None, max_length=20):
        batch_size = inputs.size(0)
        if states is None:
            states = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(inputs.device)

        generated_ids = []
        inputs = inputs.unsqueeze(1)
        for _ in range(max_length):
            rnn_output, states = self.rnn(inputs, states)
            outputs = self.fc(rnn_output.squeeze(1))
            predicted_token = outputs.argmax(dim=1)
            generated_ids.append(predicted_token.item())
            if predicted_token == 1:  # Assuming 1 is the <end> token
                break
            inputs = self.embedding(predicted_token).unsqueeze(1)
        return generated_ids


#  training
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
#
images_dir = '../COCO_Data/train2017'
captions_path = '../COCO_Data/annotations/captions_train2017.json'

dataloader = get_data_loader(
    images_dir=images_dir,
    captions_path=captions_path,
    vocab_exists=False,
    batch_size=64,
    transform=transform
)

embedding_dim = 256
hidden_dim = 512
vocab_size = len(vocab)
encoder = CustomCNNEncoder(embedding_dim).to(device)
decoder = CustomRNNDecoder(embedding_dim, hidden_dim, vocab_size).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<pad>'])
encoder_optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=5e-3)
encoder_scheduler = lr_scheduler.StepLR(encoder_optimizer, step_size=5, gamma=0.1)
decoder_scheduler = lr_scheduler.StepLR(decoder_optimizer, step_size=5, gamma=0.1)
#
# num_epochs = 40
# for epoch in range(num_epochs):
#     encoder.train()
#     decoder.train()
#     total_loss = 0
#
#     for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
#         images = batch['images'].to(device)
#         captions = batch['tokenized_caption'].to(device)
#         encoder_optimizer.zero_grad()
#         decoder_optimizer.zero_grad()
#         image_features = encoder(images)
#         outputs = decoder(image_features, captions)
#         loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
#         total_loss += loss.item()
#         loss.backward()
#         encoder_optimizer.step()
#         decoder_optimizer.step()
#
#     avg_loss = total_loss / len(dataloader)
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
#     encoder_scheduler.step()
#     decoder_scheduler.step()
#
# torch.save(encoder.state_dict(), 'encoder.pth')
# torch.save(decoder.state_dict(), 'decoder.pth')
#
#
# ####################################################################################################
# #Evaluating the model
# ####################################################################################################
# encoder.load_state_dict(torch.load('encoder.pth'))
# decoder.load_state_dict(torch.load('decoder.pth'))
#
# encoder.eval()
# decoder.eval()
#
# def generate_caption(image, max_length=200):
#     image = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         image_features = encoder(image)
#     generated_ids = decoder.generate_caption(image_features, max_length=max_length)
#     generated_caption = ' '.join([vocab.idx2word[idx] for idx in generated_ids])
#     return generated_caption
#
# # def evaluate_model(dataloader, max_length=20):
# #     references = []
# #     hypotheses = []
# #
# #     for batch in tqdm(dataloader, desc='Evaluating', unit='batch'):
# #         images = batch['images'].to(device)
# #         captions = batch['all_captions']
# #         with torch.no_grad():
# #             image_features = encoder(images)
# #             generated_ids = decoder.generate_caption(image_features, max_length=max_length)
# #             generated_caption = [vocab.idx2word[idx] for idx in generated_ids]
# #             print(generated_caption)
# #         # Append references and hypotheses for BLEU score calculation
# #         references.append(captions)
# #         hypotheses.append(generated_caption)
# #     bleu_score = corpus_bleu(references, hypotheses)
# #     return bleu_score
#
# #matchs only one trained caption with the generated caption
# def evaluate_model(dataloader, max_length=20):
#     references = []
#     hypotheses = []
#
#     for batch in tqdm(dataloader, desc='Evaluating', unit='batch'):
#         images = batch['images'].to(device)
#         captions = batch['all_captions']
#         first_reference = nltk.tokenize.word_tokenize(' '.join(captions[0]).lower())
#         references.append([first_reference])
#         with torch.no_grad():
#             image_features = encoder(images)
#             generated_ids = decoder.generate_caption(image_features, max_length=max_length)
#             generated_caption = [vocab.idx2word[idx] for idx in generated_ids]
#         hypotheses.append(generated_caption)
#     bleu_score = corpus_bleu(references, hypotheses)
#     return bleu_score
#
#
#
# images_dir = '../COCO_Data/val2017'
# captions_path = '../COCO_Data/annotations/captions_val2017.json'
# dataloader = get_data_loader(images_dir, captions_path, vocab_exists=True, batch_size=1, transform=transform)
#
# avg_bleu_score = evaluate_model(dataloader)
# print(f"Average BLEU Score: {avg_bleu_score}")



#this is to generate caption for a single image

#%%
import torch
from torchvision import transforms
from PIL import Image
from model import ImageEncoder, CaptionDecoder
from vocabulary import Vocabulary


vocab = Vocabulary(annotations_file='../COCO_Data/annotations/captions_train2017.json', vocab_exists=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def generate_caption(image_path, max_length=20):
    # Load the models
    embedding_dim = 256
    hidden_dim = 512
    vocab_size = len(vocab)
    encoder = CustomCNNEncoder(embedding_dim).to(device)
    decoder = CustomRNNDecoder(embedding_dim, hidden_dim, vocab_size).to(device)
    encoder.load_state_dict(torch.load('encoder.pth'))
    decoder.load_state_dict(torch.load('decoder.pth'))
    encoder.eval()
    decoder.eval()
    image = Image.open(image_path).convert('RGB')
    try:
        image = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error during transform: {e}")
        return ""
    with torch.no_grad():
        image_features = encoder(image)
    # Generate caption
    generated_ids = decoder.generate_caption(image_features, max_length=max_length)
    generated_caption = ' '.join([vocab.idx2word[idx] for idx in generated_ids])

    return generated_caption


image_path = '/home/ubuntu/DL/COCO_Data/test2017/000000002294.jpg'
caption = generate_caption(image_path)
print(f'Generated Caption: {caption}')
