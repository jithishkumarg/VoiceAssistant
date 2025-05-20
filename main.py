from utils.document_reader import read_pdf
from utils.speech_input import record_audio, transcribe
from utils.text_generator import ask_question
from utils.tts_output import speak
from pathlib import Path

# Load document
doc_text = read_pdf(rf"files\example.pdf")

# Record voice
audio_path = record_audio(duration=6)
with open(audio_path, 'rb') as f:
    print("File opened successfully, size:", len(f.read()))
question = transcribe(audio_path)

print(f"You asked: {question}")

# Generate response
answer = ask_question(doc_text, question)
print(f"AI says: {answer}")

# Speak it
speak(answer)
