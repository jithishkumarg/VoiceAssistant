from TTS.api import TTS
from playsound import playsound  # Cross-platform audio playback
import winsound

# Load the TTS model (you can set gpu=False if you don't have a GPU)
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=False)

def speak(text):
    output_path = "response.wav"
    tts.tts_to_file(text=text, file_path=output_path)
    playsound(output_path)
    winsound.PlaySound(output_path, winsound.SND_FILENAME)
