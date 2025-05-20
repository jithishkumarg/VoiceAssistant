import whisper
import os
import tempfile

model = whisper.load_model("base")
file = r"C:\Users\Dell\output.wav"  # copy actual file path


print("Default temp dir:", tempfile.gettempdir())


print("Exists:", os.path.exists(file))
try:
    print("Transcribe test:", model.transcribe(str(file))["text"])
except Exception as e:
    print("failed to trnascribe", e)
