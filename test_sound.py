

# import pyttsx3
# import pygame
# import os

# def text_to_speech_with_playback(text, filename="output.mp3"):
#     # Initialize the pyttsx3 engine
#     engine = pyttsx3.init()

#     # Save speech to an MP3 file
#     engine.save_to_file(text, filename)
#     engine.runAndWait()

#     print(f"Speech saved as {filename}")
#     # Verify file creation
#     if not os.path.exists(filename):
#         print(f"Error: {filename} not found.")
#         return

#     # Play the generated audio file using pygame
#     pygame.mixer.init()
#     pygame.mixer.music.load(filename)
#     pygame.mixer.music.play()
#     while pygame.mixer.music.get_busy():
#         continue

# # Example text to convert
# example_text = "Hello! This is a test for text-to-speech using pyttsx3 and pygame."
# text_to_speech_with_playback(example_text)



from gtts import gTTS
import os

def text_to_speech_gtts(text, filename="output.mp3"):
    # Convert text to speech
    tts = gTTS(text)
    tts.save(filename)
    print(f"Speech saved as {filename}")
    # Play the audio file
    os.system(f"mpg123 {filename}")

example_text = "This is a test using gTTS for text-to-speech conversion."
text_to_speech_gtts(example_text)



# from gtts import gTTS
# import os

# def text_to_speech_gtts(text, filename="output.mp3", lang="en"):
#     """
#     Converts text to speech using gTTS with the specified language.
    
#     Args:
#         text (str): Text to convert to speech.
#         filename (str): File name to save the audio file.
#         lang (str): Language for the text-to-speech conversion (default is 'en').
#     """
#     # Convert text to speech in the specified language
#     tts = gTTS(text, lang=lang)
#     tts.save(filename)
#     print(f"Speech saved as {filename}")
#     # Play the audio file
#     os.system(f"mpg123 {filename}")

# # Example text
# example_text = "Hello! How are you?"

# # Convert to French
# text_to_speech_gtts(example_text, filename="output_french.mp3", lang="fr")

# # Convert to Spanish
# text_to_speech_gtts(example_text, filename="output_spanish.mp3", lang="es")

# # Convert to Hindi
# text_to_speech_gtts(example_text, filename="output_hindi.mp3", lang="hi")



#######  TRANSLATIONNN ################

# from gtts import gTTS
# from deep_translator import GoogleTranslator
# import os

# def translate_and_speak(text, target_lang, filename="output.mp3"):
#     """
#     Translates text to the target language and converts it to speech.
    
#     Args:
#         text (str): Text to translate and convert.
#         target_lang (str): Target language code (e.g., "hi" for Hindi).
#         filename (str): File to save the audio.
#     """
#     # Translate the text
#     translated = GoogleTranslator(source="auto", target=target_lang).translate(text)
#     print(f"Translated Text: {translated}")
    
#     # Convert translated text to speech
#     tts = gTTS(translated, lang=target_lang)
#     tts.save(filename)
#     print(f"Speech saved as {filename}")
#     os.system(f"mpg123 {filename}")

# # Example usage
# example_text = "Hello! How are you?"
# translate_and_speak(example_text, target_lang="hi", filename="output_hindi.mp3")



from gtts import gTTS
from deep_translator import GoogleTranslator
import os

def translate_and_speak(text, target_lang, filename="output.mp3"):
   
    translated = GoogleTranslator(source="auto", target=target_lang).translate(text)
    print(f"Translated Text: {translated}")
    
    tts = gTTS(translated, lang=target_lang)
    tts.save(filename)
    print(f"Speech saved as {filename}")
    os.system(f"mpg123 {filename}")


example_text = "Hello! How are you?"


translate_and_speak(example_text, target_lang="hi", filename="output_hindi.mp3")


translate_and_speak(example_text, target_lang="es", filename="output_spanish.mp3")


translate_and_speak(example_text, target_lang="fr", filename="output_french.mp3")



