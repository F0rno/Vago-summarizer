import whisper
import logging

class AudioTranscriber:
	def __init__(self, audio_path):
		self.audio_path = audio_path
		self.model = whisper.load_model("base")

	def transcribe(self):
		logging.info(f"Transcribing audio file {self.audio_path}")
		result = self.model.transcribe(self.audio_path)
		text = result["text"]
		return text