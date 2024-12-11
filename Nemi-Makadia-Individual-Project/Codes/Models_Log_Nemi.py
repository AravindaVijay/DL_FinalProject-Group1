# Summary of Models Tried in the Project

# Model 1: ResNet50 + LSTM (Baseline Model)
from torchvision import models
import torch.nn as nn

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
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
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

# Model 2: Vision Transformer (ViT) + Transformer Decoder
from transformers import ViTModel

class ViTEncoder(nn.Module):
    def __init__(self, embed_size):
        super(ViTEncoder, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.fc = nn.Linear(self.vit.config.hidden_size, embed_size)

    def forward(self, images):
        outputs = self.vit(pixel_values=images)
        features = self.fc(outputs.last_hidden_state[:, 0, :])
        return features

class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer = nn.Transformer(d_model=embed_size, nhead=8, num_encoder_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions[:, :-1])
        embeddings = embeddings.permute(1, 0, 2)
        features = features.unsqueeze(0)
        outputs = self.transformer(features, embeddings)
        outputs = self.fc(outputs)
        return outputs

# Model 3: Custom CNN + RNN
class CustomCNN(nn.Module):
    def __init__(self, embed_size):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 112 * 112, embed_size)

    def forward(self, images):
        x = self.conv1(images)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CustomRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(CustomRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.rnn(embeddings)
        outputs = self.fc(hiddens)
        return outputs

# Model 4: Vision Transformer Encoder + LSTM Decoder
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

# Model 5: BLIP-2
from transformers import BlipProcessor, BlipForConditionalGeneration

class BLIP2Model:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    def generate_captions(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        captions = [self.processor.decode(output, skip_special_tokens=True) for output in outputs]
        return captions




############################################# APP.PY
# GET CODE FOR DIFFERENT LANGUAGE BUTTONS AND ADD
    
# import streamlit as st
# from gtts import gTTS
# from io import BytesIO
# from PIL import Image

# #css
# st.markdown(
#     """
#     <style>
#         .header-container {
#             display: flex;
#             align-items: center;
#             padding-top: 20px;
#             padding-bottom: 20px;
#             margin-bottom: 20px; 
#             justify-content: center;
#         }
#         .header-title {
#             font-size: 4.8em;
#             font-weight: bold;
#             margin-left: 20px;
#             font-family: 'Algerian', sans-serif;  
#         }
#     </style>
#     """, 
#     unsafe_allow_html=True
# )


# def home_demo_page():
#     st.markdown('<div class="header-container">', unsafe_allow_html=True)
#     logo = Image.open("assets/logo.png")  
#     col1, col2 = st.columns([2, 5])  
#     with col1:
#         st.image(logo, width=500)  
#     with col2:
#         st.markdown('<div class="header-title" style="text-align: center;">Vision Voice</div>', unsafe_allow_html=True)
#         st.markdown('<h3 class="sub" style="text-align: center; font-family: Algerian, sans-serif;">Hear the image</h3>', unsafe_allow_html=True)
        
    
#     st.markdown('</div>', unsafe_allow_html=True)

#     st.write("Welcome to the Image Captioning App!")

#     st.header("Demo")
#     uploaded_file = st.file_uploader("Drag and drop an image here", type=["jpg", "jpeg", "png"])
    
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Image", use_column_width=True)
#         caption = "This is a sample caption for the uploaded image."
#         st.write("Generated Caption: ", caption)
#         if st.button("🔊 Convert to Speech"):
#             tts = gTTS(text=caption, lang='en')
#             audio_bytes = BytesIO()
#             tts.write_to_fp(audio_bytes)
#             st.audio(audio_bytes, format="audio/mp3")

# def about_page():
#     st.header("About the Project")
#     st.write("""
#         This project uses deep learning to generate captions for images.
#         The model is trained to understand images and provide descriptive captions.
#         The app also allows converting these captions to audio.
#     """)

# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Go to", ("Home/Demo", "About"))

# if page == "Home/Demo":
#     home_demo_page()
# elif page == "About":
#     about_page()
    


###############################################################################
    ########### WHOLE MODEL.PY CODE
###############################################################################
    
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
# class ImageEncoder(nn.Module):
#     def __init__(self, embedding_dim, trainable_layers=0):
#         super(ImageEncoder, self).__init__()
#         base_model = models.resnet50(pretrained=True)
#         for param in base_model.parameters():
#             param.requires_grad = False
#         for param in list(base_model.parameters())[-trainable_layers:]:
#             param.requires_grad = True
#         feature_extractor = list(base_model.children())[:-1]
#         self.feature_extractor = nn.Sequential(*feature_extractor)
#         self.fc = nn.Linear(base_model.fc.in_features, embedding_dim)
#
#     def forward(self, images):
#         extracted_features = self.feature_extractor(images)
#         extracted_features = extracted_features.view(extracted_features.size(0), -1)
#         embeddings = self.fc(extracted_features)
#         return embeddings

class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim, trainable_layers=0):
        super(ImageEncoder, self).__init__()
        base_model = models.efficientnet_b3(pretrained=True)
        for param in base_model.parameters():
            param.requires_grad = False
        if trainable_layers > 0:
            feature_params = list(base_model.features.parameters())
            for param in feature_params[-trainable_layers:]:
                param.requires_grad = True
        self.feature_extractor = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1536, embedding_dim)

    def forward(self, images):
        x = self.feature_extractor(images)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        embeddings = self.fc(x)
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
        self.vocab = vocab
        self.start_token_id = vocab(vocab.start_word)

    def forward(self, image_features, captions):
        caption_embeddings = self.word_embeddings(captions[:, :-1])
        inputs = torch.cat((image_features.unsqueeze(1), caption_embeddings), dim=1)
        lstm_output, self.hidden_state = self.lstm(inputs)
        outputs = self.fc(lstm_output)
        return outputs

    # def generate_caption(self, inputs, states=None, max_length=20):
    #     """
    #     Generate a caption for a given image feature vector.
    #     Args:
    #         inputs: Image feature vector (shape: [batch_size, embedding_dim]).
    #         states: Initial hidden and cell states for LSTM.
    #         max_length: Maximum length of the generated caption.
    #     Returns:
    #         generated_ids: List of word indices forming the generated caption.
    #     """
    #     batch_size = inputs.size(0)  # Get the batch size
    #     if states is None:
    #         states = (
    #             torch.zeros(1, batch_size, self.hidden_dim).to(inputs.device),
    #             torch.zeros(1, batch_size, self.hidden_dim).to(inputs.device)
    #         )
    #     generated_ids = []
    #     inputs = inputs.unsqueeze(1)
    #
    #     for _ in range(max_length):
    #         lstm_output, states = self.lstm(inputs, states)
    #         outputs = self.fc(lstm_output.squeeze(1))
    #         predicted_token = outputs.argmax(dim=1)
    #         generated_ids.append(predicted_token.item())
    #         if predicted_token.item() == self.end_token_id:
    #             break
    #         inputs = self.word_embeddings(predicted_token).unsqueeze(1)
    #     return generated_ids

    #beam search captions
    def generate_caption(self, image_features, beam_size=3, max_length=20):
        device = image_features.device
        vocab_size = self.fc.out_features
        assert image_features.size(0) == 1, "Batch size should be 1 for inference."
        start_token_id = self.start_token_id
        end_token_id = self.end_token_id
        h = torch.zeros(self.lstm.num_layers, beam_size, self.hidden_dim, device=device)
        c = torch.zeros(self.lstm.num_layers, beam_size, self.hidden_dim, device=device)
        seqs = torch.LongTensor([[start_token_id]]).to(device)
        seqs = seqs.repeat(beam_size, 1)
        top_k_scores = torch.zeros(beam_size, 1, device=device)
        image_features = image_features.repeat(beam_size, 1)
        image_features = image_features.unsqueeze(1)
        with torch.no_grad():
            lstm_output, (h, c) = self.lstm(image_features, (h, c))
        complete_seqs = []
        complete_seqs_scores = []
        for step in range(max_length):
            last_word_ids = seqs[:, -1]
            embeddings = self.word_embeddings(last_word_ids)
            embeddings = embeddings.unsqueeze(1)
            lstm_output, (h, c) = self.lstm(embeddings, (h, c))
            scores = self.fc(lstm_output.squeeze(1))
            scores = F.log_softmax(scores, dim=1)
            scores = top_k_scores.expand_as(scores) + scores
            top_k_scores_flat, top_k_words_flat = scores.view(-1).topk(beam_size, dim=0, largest=True, sorted=True)
            prev_word_inds = top_k_words_flat // vocab_size
            next_word_inds = top_k_words_flat % vocab_size
            new_seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
            new_scores = top_k_scores_flat
            complete_inds = (next_word_inds == end_token_id).nonzero(as_tuple=False).squeeze(1)
            incomplete_inds = (next_word_inds != end_token_id).nonzero(as_tuple=False).squeeze(1)
            if len(complete_inds) > 0:
                for i in complete_inds:
                    complete_seqs.append(new_seqs[i].tolist())
                    complete_seqs_scores.append(new_scores[i].item())
            k = beam_size - len(complete_seqs)
            if k <= 0:
                break
            seqs = new_seqs[incomplete_inds]
            top_k_scores = new_scores[incomplete_inds].unsqueeze(1)

            h = h[:, prev_word_inds[incomplete_inds], :]
            c = c[:, prev_word_inds[incomplete_inds], :]

        if len(complete_seqs_scores) == 0:
            complete_seqs = seqs.tolist()
            complete_seqs_scores = top_k_scores.squeeze(1).tolist()


        best_idx = complete_seqs_scores.index(max(complete_seqs_scores))
        best_seq = complete_seqs[best_idx]

        words = [self.vocab.idx2word[idx] for idx in best_seq]
        # Remove the <end> token if it exists
        if end_token_id in best_seq:
            words = words[:best_seq.index(end_token_id)]

        return words


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


def main():

    vocab = Vocabulary(annotations_file='../COCO_Data/annotations/captions_train2017.json', vocab_exists=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # class CustomCNNEncoder(nn.Module):
    #     def __init__(self, embedding_dim):
    #         super(CustomCNNEncoder, self).__init__()
    #         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)  # Downscale image
    #         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
    #         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
    #         self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Pool to 1x1 feature map
    #         self.fc = nn.Linear(64, embedding_dim)
    #
    #     def forward(self, images):
    #         x = torch.relu(self.conv1(images))
    #         x = torch.relu(self.conv2(x))
    #         x = torch.relu(self.conv3(x))
    #         x = self.pool(x)
    #         x = x.view(x.size(0), -1)  # Flatten
    #         x = self.fc(x)
    #         return x
    #
    # 
    # class CustomRNNDecoder(nn.Module):
    #     def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers=1):
    #         super(CustomRNNDecoder, self).__init__()
    #         self.embedding = nn.Embedding(vocab_size, embedding_dim)
    #         self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
    #         self.fc = nn.Linear(hidden_dim, vocab_size)
    #         self.hidden_dim = hidden_dim
    #         self.num_layers = num_layers
    #
    #     def forward(self, image_features, captions):
    #         embeddings = self.embedding(captions[:, :-1])  # Exclude <end> token
    #         inputs = torch.cat((image_features.unsqueeze(1), embeddings), dim=1)
    #         rnn_output, _ = self.rnn(inputs)
    #         outputs = self.fc(rnn_output)
    #         return outputs
    #
    #     def generate_caption(self, inputs, states=None, max_length=20):
    #         batch_size = inputs.size(0)
    #         if states is None:
    #             states = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(inputs.device)
    #
    #         generated_ids = []
    #         inputs = inputs.unsqueeze(1)
    #         for _ in range(max_length):
    #             rnn_output, states = self.rnn(inputs, states)
    #             outputs = self.fc(rnn_output.squeeze(1))
    #             predicted_token = outputs.argmax(dim=1)
    #             generated_ids.append(predicted_token.item())
    #             if predicted_token == 1:  # Assuming 1 is the <end> token
    #                 break
    #             inputs = self.embedding(predicted_token).unsqueeze(1)
    #         return generated_ids




    #  training
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
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
        batch_size=128,
        transform=transform
    )

    embedding_dim = 256
    hidden_dim = 512
    vocab_size = len(vocab)
    encoder = ImageEncoder(embedding_dim).to(device)
    decoder = CaptionDecoder(embedding_dim, hidden_dim, vocab_size, vocab).to(device)

    criterion = (
        nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    )
    params = list(decoder.parameters()) + list(encoder.fc.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)
    encoder_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    decoder_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    #
    num_epochs = 10
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
            images = batch['images'].to(device)
            captions = batch['tokenized_caption'].to(device)

            encoder.zero_grad()
            decoder.zero_grad()

            image_features = encoder(images)
            outputs = decoder(image_features, captions)
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        encoder_scheduler.step()
        decoder_scheduler.step()

    torch.save(encoder.state_dict(), 'encoder.pth')
    torch.save(decoder.state_dict(), 'decoder.pth')


    ####################################################################################################
    #Evaluating the model
    ####################################################################################################

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

    encoder = ImageEncoder(embedding_dim).to(device)
    decoder = CaptionDecoder(embedding_dim, hidden_dim, vocab_size, vocab).to(device)

    encoder.load_state_dict(torch.load('encoder.pth'))
    decoder.load_state_dict(torch.load('decoder.pth'))

    encoder.eval()
    decoder.eval()

    def generate_caption(image, max_length=200):
        image = transform_test(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = encoder(image)
        generated_ids = decoder.generate_caption(image_features, max_length=max_length)
        generated_caption = ' '.join([vocab.idx2word[idx] for idx in generated_ids])
        return generated_caption


    #matchs only one trained caption with the generated caption
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

    # def evaluate_model(dataloader, max_length=20):
    #     true_sentences = {}
    #     predicted_sentences = {}
    #
    #     for batch in tqdm(dataloader, desc='Evaluating', unit='batch'):
    #         images = batch['images'].to(device)
    #         captions = batch['all_captions']
    #         image_ids = batch['image_ids']
    #
    #         for i, img_id in enumerate(image_ids):
    #             true_sentences[img_id] = captions[i]
    #
    #         with torch.no_grad():
    #             image_features = encoder(images)
    #             generated_ids = decoder.generate_caption(image_features, max_length=max_length)
    #             generated_caption = ' '.join([vocab.idx2word[idx] for idx in generated_ids])
    #             for i, img_id in enumerate(image_ids):
    #                 predicted_sentences[img_id] = [generated_caption]
    #
    #     avg_bleu_score = bleu_score(true_sentences, predicted_sentences)
    #     return avg_bleu_score

    #beam search generate caption
    def evaluate_model(dataloader, max_length=20):
        true_sentences = {}
        predicted_sentences = {}

        for batch in tqdm(dataloader, desc='Evaluating', unit='batch'):
            images = batch['images'].to(device)
            captions = batch['all_captions']
            image_ids = batch['image_ids']

            for i, img_id in enumerate(image_ids):
                true_sentences[img_id] = captions[i]

            with torch.no_grad():
                image_features = encoder(images)  # Assuming batch_size=1
                generated_words = decoder.generate_caption(image_features, beam_size=3, max_length=max_length)
                generated_caption = ' '.join(generated_words)
                for i, img_id in enumerate(image_ids):
                    predicted_sentences[img_id] = [generated_caption]

        avg_bleu_score = bleu_score(true_sentences, predicted_sentences)
        return avg_bleu_score


    def bleu_score(true_sentences, predicted_sentences):
        hypotheses = []
        references = []
        for img_id in set(true_sentences.keys()).intersection(set(predicted_sentences.keys())):
            img_refs = [cap.split() for cap in true_sentences[img_id]]
            references.append(img_refs)
            hypotheses.append(predicted_sentences[img_id][0].strip().split())

        return corpus_bleu(references, hypotheses)



    images_dir = '../COCO_Data/val2017'
    captions_path = '../COCO_Data/annotations/captions_val2017.json'
    dataloader = get_data_loader(images_dir, captions_path, vocab_exists=True, batch_size=1, transform=transform)

    avg_bleu_score = evaluate_model(dataloader)
    print(f"Average BLEU Score: {avg_bleu_score}")


    ##########################################################################################################################
    #this is to generate caption for a single image
    ############################################################################################################################
    # # %%
    # import os
    # os.chdir('/home/ubuntu/DL/Code')
    # import torch
    # from torchvision import transforms
    # from PIL import Image
    # from model import ImageEncoder, CaptionDecoder
    # from vocabulary import Vocabulary
    #
    #
    # vocab = Vocabulary(annotations_file='../COCO_Data/annotations/captions_train2017.json', vocab_exists=True)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # transform = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    # #
    # def generate_caption(image_path, max_length=20):
    #     # Load the models
    #     embedding_dim = 256
    #     hidden_dim = 512
    #     vocab_size = len(vocab)
    #     encoder = ImageEncoder(embedding_dim).to(device)
    #     decoder = CaptionDecoder(embedding_dim, hidden_dim, vocab_size, vocab).to(device)
    #     encoder.load_state_dict(torch.load('encoder.pth'))
    #     decoder.load_state_dict(torch.load('decoder.pth'))
    #     encoder.eval()
    #     decoder.eval()
    #     image = Image.open(image_path).convert('RGB')
    #     try:
    #         image = transform(image).unsqueeze(0).to(device)
    #     except Exception as e:
    #         print(f"Error during transform: {e}")
    #         return ""
    #     with torch.no_grad():
    #         image_features = encoder(image)
    #     # Generate caption
    #     generated_ids = decoder.generate_caption(image_features, max_length=max_length)
    #     generated_caption = ' '.join([vocab.idx2word[idx] for idx in generated_ids])
    #
    #     return generated_caption
    #
    # #use this for beam search generate
    # def generatecaption(image_path, beam_size=3, max_length=20):
    #     # Load the models
    #     embedding_dim = 256
    #     hidden_dim = 512
    #     vocab_size = len(vocab)
    #     encoder = ImageEncoder(embedding_dim).to(device)
    #     decoder = CaptionDecoder(embedding_dim, hidden_dim, vocab_size, vocab).to(device)
    #     encoder.load_state_dict(torch.load('encoder.pth'))
    #     decoder.load_state_dict(torch.load('decoder.pth'))
    #     encoder.eval()
    #     decoder.eval()
    #     image = Image.open(image_path).convert('RGB')
    #     try:
    #         image = transform(image).unsqueeze(0).to(device)
    #     except Exception as e:
    #         print(f"Error during transform: {e}")
    #         return ""
    #     with torch.no_grad():
    #         image_features = encoder(image)
    #
    #     # Generate caption using beam search
    #     generated_words = decoder.generate_caption(image_features, beam_size=beam_size, max_length=max_length)
    #     generated_caption = ' '.join(generated_words)
    #
    #     return generated_caption
    #
    #
    # image_path = '//home/ubuntu/DL/COCO_Data/test2017/000000004432.jpg'
    # caption = generate_caption(image_path)
    # print(f'Generated Caption: {caption}')

#%%

if __name__ == '__main__':
    main()
