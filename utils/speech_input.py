import whisper
import sounddevice as sd
import soundfile as sf
import tempfile
import os
import time

model = whisper.load_model("base")


def record_audio(duration=5, samplerate=44100):
    print("Listening...")
    recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1)
    sd.wait()

    # Create the temp file path, then close it
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    # Now write to it safely
    sf.write(tmp_path, recording, samplerate)
    time.sleep(0.5)
    return tmp_path

def transcribe(file_path):
    custom_path = r"C:\Users\Dell\AppData\Local\Temp\tmpekunr9bd.wav"
    print(f"Type: {type(file_path)}")
    print("Audio path:", file_path)
    print("Exists:", os.path.exists(file_path))
    return model.transcribe(file_path)["text"]
