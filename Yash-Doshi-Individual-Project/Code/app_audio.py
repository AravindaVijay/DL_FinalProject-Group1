import streamlit as st
from gtts import gTTS
from deep_translator import GoogleTranslator
from io import BytesIO
from PIL import Image


st.markdown(
    """
    <style>
        .header-container {
            display: flex;
            align-items: center;
            padding-top: 20px;
            padding-bottom: 20px;
            margin-bottom: 20px; 
            justify-content: center;
        }
        .header-title {
            font-size: 4.8em;
            font-weight: bold;
            margin-left: 20px;
            font-family: 'Algerian', sans-serif;  
        }
    </style>
    """, 
    unsafe_allow_html=True
)


def translate_and_speak(text, target_lang):
    translated = GoogleTranslator(source="auto", target=target_lang).translate(text)
    tts = gTTS(translated, lang=target_lang)
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    return translated, audio_bytes


def home_demo_page():
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="header-title" style="text-align: center;">Vision Voice</h1>', unsafe_allow_html=True)
    st.write("Welcome to the Image Captioning App!")

    st.header("Demo")
    option = st.radio("Choose an image source:", ("Upload from device", "Capture from camera"))

    image = None
    if option == "Upload from device":
        uploaded_file = st.file_uploader("Drag and drop an image here", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

    elif option == "Capture from camera":
        camera_image = st.camera_input("Take a picture")
        if camera_image is not None:
            image = Image.open(camera_image).convert("RGB")

    if image is not None:
       
        st.image(image, caption="Selected Image", use_container_width=True)

       
        caption = "This is a placeholder caption for the selected image."
        st.write("Generated Caption: ", caption)

        st.subheader("Convert Caption to Speech in Different Languages:")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("🔊 English"):
                st.write("Caption in English: ", caption)
                tts = gTTS(text=caption, lang='en')
                audio_bytes = BytesIO()
                tts.write_to_fp(audio_bytes)
                st.audio(audio_bytes, format="audio/mp3")

        with col2:
            if st.button("🔊 Hindi"):
                translated, audio_bytes = translate_and_speak(caption, target_lang="hi")
                st.write("Caption in Hindi: ", translated)
                st.audio(audio_bytes, format="audio/mp3")

        with col3:
            if st.button("🔊 Spanish"):
                translated, audio_bytes = translate_and_speak(caption, target_lang="es")
                st.write("Caption in Spanish: ", translated)
                st.audio(audio_bytes, format="audio/mp3")

        with col4:
            if st.button("🔊 French"):
                translated, audio_bytes = translate_and_speak(caption, target_lang="fr")
                st.write("Caption in French: ", translated)
                st.audio(audio_bytes, format="audio/mp3")

    else:
        st.write("Please upload or capture an image to generate a caption.")


def about_page():
    st.header("About the Project")
    st.write("""
        This project uses deep learning to generate captions for images.
        The model is trained to understand images and provide descriptive captions.
        The app also allows converting these captions to audio in multiple languages.
    """)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Home/Demo", "About"))

st.sidebar.header("Feedback")
feedback = st.sidebar.text_area("Have feedback? Let us know!")
if st.sidebar.button("Submit Feedback"):
    if feedback.strip():
        st.sidebar.success("Thank you for your feedback!")
    else:
        st.sidebar.error("Feedback cannot be empty!")

if page == "Home/Demo":
    home_demo_page()
elif page == "About":
    about_page()
