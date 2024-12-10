import streamlit as st
from gtts import gTTS
from io import BytesIO
from PIL import Image
from generate_caption_beam import generate_caption_beam
from generate_caption_resnet import generate_caption_resnet
from generate_caption_blip import generate_caption_blip

# CSS for styling
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

# Home/Demo Page
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
    # st.markdown('<div class="header-container">', unsafe_allow_html=True)
    # st.markdown('<h1 class="header-title" style="text-align: center;">Vision Voice</h1>', unsafe_allow_html=True)
    st.write("Welcome to the Image Captioning App!")

    st.header("Demo")

    model_option = st.selectbox("Choose a model for caption generation:", ("Custom CNN", "EfficientNet (Beam Search)", "Blip"))

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
        # Display the uploaded or captured image
        st.image(image, caption="Selected Image", use_container_width=True)

        # Generate caption based on selected model
        if model_option == "Custom CNN":
            caption = generate_caption_beam(image)
        elif model_option == "EfficientNet (Beam Search)":
            caption = generate_caption_resnet(image)
        elif model_option == "Blip":
            caption = generate_caption_blip(image)
        st.write("Generated Caption: ", caption)

        # Convert caption to speech
        if st.button("🔊 Convert to Speech"):
            tts = gTTS(text=caption, lang='en')
            audio_bytes = BytesIO()
            tts.write_to_fp(audio_bytes)
            st.audio(audio_bytes, format="audio/mp3")
    else:
        st.write("Please upload or capture an image to generate a caption.")

# About Page
def about_page():
    st.header("About the Project")
    st.write("""
        This project uses deep learning to generate captions for images.
        The model is trained to understand images and provide descriptive captions.
        The app also allows converting these captions to audio.
    """)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Home/Demo", "About"))

# Feedback Section
st.sidebar.header("Feedback")
feedback = st.sidebar.text_area("Have feedback? Let us know!")
if st.sidebar.button("Submit Feedback"):
    if feedback.strip():
        st.sidebar.success("Thank you for your feedback!")
    else:
        st.sidebar.error("Feedback cannot be empty!")

# Page Routing
if page == "Home/Demo":
    home_demo_page()
elif page == "About":
    about_page()






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
