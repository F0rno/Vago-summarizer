import logging
from os.path import exists
from os import environ
from video_processor import VideoProcessor
from audio_transcriber import AudioTranscriber
from api import OllamaClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_environment():
    environ["TOKENIZERS_PARALLELISM"] = "false"

def file_exists(file_path):
    return exists(file_path)

def read_file(file_path):
    try:
        with open(file_path, "r") as file:
            return file.read()
    except IOError:
        return None

def write_file(file_path, content):
    try:
        with open(file_path, "w") as file:
            file.write(content)
    except IOError as e:
        logging.error(f"Error writing to file {file_path}: {e}")

def summarize_chunks(prompt, system_prompt, api: OllamaClient):
    return api.call_llm(prompt, system_prompt)

def extract_audio_if_needed(video_path, audio_path):
    if not file_exists(audio_path):
        VideoProcessor(video_path, audio_path).extract_audio()

def transcribe_audio_if_needed(audio_path, transcript_path):
    if not file_exists(transcript_path):
        transcript = AudioTranscriber(audio_path).transcribe()
        write_file(transcript_path, transcript)
    else:
        transcript = read_file(transcript_path)
    return transcript
