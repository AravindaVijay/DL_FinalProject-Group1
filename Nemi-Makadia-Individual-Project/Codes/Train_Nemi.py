##########################################################################
######################PRETRAINED RESNET AND LSTM########################
##########################################################################

# ResNet50 + LSTM (Baseline Model) (Tried the training loop on this model)


# class EncoderCNN(nn.Module):
#     def __init__(self, embed_size):
#         super(EncoderCNN, self).__init__()
#         resnet = models.resnet50(pretrained=True)
#         for param in resnet.parameters():
#             param.requires_grad_(False)
#         self.resnet = nn.Sequential(*list(resnet.children())[:-1])
#         self.embed = nn.Linear(resnet.fc.in_features, embed_size)

#     def forward(self, images):
#         features = self.resnet(images)
#         features = features.view(features.size(0), -1)
#         features = self.embed(features)
#         return features

# class DecoderRNN(nn.Module):
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
#         super(DecoderRNN, self).__init__()
#         self.embed = nn.Embedding(vocab_size, embed_size)
#         self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
#         self.linear = nn.Linear(hidden_size, vocab_size)

#     def forward(self, features, captions):
#         embeddings = self.embed(captions[:, :-1])
#         embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
#         hiddens, _ = self.lstm(embeddings)
#         outputs = self.linear(hiddens)
#         return outputs


# TRAINING LOOP (USED TO TRAIN A BASELINE MODEL INTEGRATED BY MY GROUPMATE)

from torchvision import models
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu

print("Starting training...")
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()

    total_loss = 0  

    for i, batch in enumerate(dataloader):
        images = batch['images'].to(device)
        captions = batch['tokenized_caption'].to(device)

        features = encoder(images)
        outputs = decoder(features, captions)

        loss = criterion(outputs.view(-1, len(dataloader.dataset.vocab)), captions.view(-1))
        total_loss += loss.item()

        optimizer.zero_grad()   
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    torch.save(encoder.state_dict(), f"encoder_epoch_{epoch+1}.pth") # saving each epoch's model
    torch.save(decoder.state_dict(), f"decoder_epoch_{epoch+1}.pth")
    print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {total_loss / len(dataloader):.4f}. Models saved.")

    # Evaluation 
    encoder.eval()
    decoder.eval()

    references = [] #ground truth captions
    hypotheses = []  #generated captions

    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            captions = batch['tokenized_caption']

            features = encoder(images)
            generated_ids = decoder.generate_caption(features)

            generated_caption = [dataloader.dataset.vocab.idx2word[idx] for idx in generated_ids]  # tokenized captions to words
            reference_caption = [[dataloader.dataset.vocab.idx2word[idx] for idx in caption if idx not in {0, 1, 2}] for caption in captions]

            hypotheses.append(generated_caption)
            references.append(reference_caption)

    bleu_score = corpus_bleu(references, hypotheses)
    print(f"Epoch {epoch+1} BLEU Score: {bleu_score:.4f}")

print("Training completed successfully!")




# Tried to add a few features in the app.
    
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
    