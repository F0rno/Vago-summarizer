from video_processor import VideoProcessor
from audio_transcriber import AudioTranscriber

if __name__ == "__main__":
    video_path = "index.mp4"
    video_processor = VideoProcessor(video_path)
    audio_path = video_processor.extract_audio()
    
    audio_transcriber = AudioTranscriber(audio_path)
    audio_transcriber.transcribe()