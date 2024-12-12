
#####################################################
####### TTS Module #################################
#####################################################

from gtts import gTTS
import os

def text_to_speech_gtts(text, filename="output.mp3"):

    tts = gTTS(text) #text to speech
    tts.save(filename)
    print(f"Speech saved as {filename}")
    os.system(f"mpg123 {filename}")

example_text = "This is a test using gTTS for text-to-speech conversion."
text_to_speech_gtts(example_text)

def text_to_speech_gtts(text, filename="output.mp3", lang="en"):
    """
    Converts text to speech using gTTS with the specified language.
    
    Args:
        text (str): Text to convert to speech.
        filename (str): File name to save the audio file.
        lang (str): Language for the text-to-speech conversion (default is 'en').
    """
    tts = gTTS(text, lang=lang)
    tts.save(filename)
    print(f"Speech saved as {filename}")
    # Play the audio file
    os.system(f"mpg123 {filename}")

example_text = "Hello! How are you?"

text_to_speech_gtts(example_text, filename="output_french.mp3", lang="fr")
text_to_speech_gtts(example_text, filename="output_spanish.mp3", lang="es")
text_to_speech_gtts(example_text, filename="output_hindi.mp3", lang="hi")





