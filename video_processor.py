import logging
import moviepy.editor as mp

class VideoProcessor:
	def __init__(self, video_path, output_audio_path):
		self.video_path = video_path
		self.output_audio_path = output_audio_path

	def extract_audio(self):
		logging.info("Extracting the audio from the video...")
		video = mp.VideoFileClip(self.video_path)
		audio = video.audio
		audio.write_audiofile(self.output_audio_path)
