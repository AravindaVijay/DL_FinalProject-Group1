import streamlit as st
from gtts import gTTS
from io import BytesIO
from PIL import Image
from generate_caption import generate_caption

# supress all warnings
import warnings
warnings.filterwarnings("ignore")

#css
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


def home_demo_page():
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    logo = Image.open("assets/logo.png")  
    col1, col2 = st.columns([2, 5])  
    with col1:
        st.image(logo, width=500)  
    with col2:
        st.markdown('<div class="header-title" style="text-align: center;">Vision Voice</div>', unsafe_allow_html=True)
        st.markdown('<h3 class="sub" style="text-align: center; font-family: Algerian, sans-serif;">Hear the image</h3>', unsafe_allow_html=True)
        
    
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("Welcome to the Image Captioning App!")

    st.header("Demo")
    uploaded_file = st.file_uploader("Drag and drop an image here", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        caption = generate_caption(image)
        # caption = "Sample caption"
        st.write("Generated Caption: ", caption)
        if st.button("🔊 Convert to Speech"):
            tts = gTTS(text=caption, lang='en')
            audio_bytes = BytesIO()
            tts.write_to_fp(audio_bytes)
            st.audio(audio_bytes, format="audio/mp3")

def about_page():
    st.header("About the Project")
    st.write("""
        This project uses deep learning to generate captions for images.
        The model is trained to understand images and provide descriptive captions.
        The app also allows converting these captions to audio.
    """)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Home/Demo", "About"))

if page == "Home/Demo":
    home_demo_page()
elif page == "About":
    about_page()



