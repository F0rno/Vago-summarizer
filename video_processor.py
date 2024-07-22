import moviepy.editor as mp
import os

class VideoProcessor:
	def __init__(self, video_path):
		self.video_path = video_path
		self.base_name = os.path.splitext(os.path.basename(video_path))[0]
		self.output_audio_path = f"./{self.base_name}.mp3"

	def extract_audio(self):
		if not os.path.exists(self.output_audio_path):
			video = mp.VideoFileClip(self.video_path)
			audio = video.audio
			audio.write_audiofile(self.output_audio_path)
			print(f"Audio extracted and saved to {self.output_audio_path}")
		else:
			print(f"Audio file {self.output_audio_path} already exists. Skipping extraction.")
		return self.output_audio_path