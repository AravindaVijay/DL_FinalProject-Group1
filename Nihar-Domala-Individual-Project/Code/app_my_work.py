import streamlit as st
from gtts import gTTS
from deep_translator import GoogleTranslator
from io import BytesIO
from PIL import Image
from generate_caption_beam import generate_caption_beam
from generate_caption_resnet import generate_caption_resnet
from generate_caption_blip import generate_caption_blip



model_option = st.selectbox("Choose a model for caption generation:", ("ResNet" ,"EfficientNet", "Blip"))

# Generate caption based on selected model
if model_option == "ResNet":
    caption = generate_caption_resnet(image)
elif model_option == "EfficientNet":
    caption = generate_caption_beam(image)
elif model_option == "Blip":
    caption = generate_caption_blip(image)
st.write("Generated Caption: ", caption)