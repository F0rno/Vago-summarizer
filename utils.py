import logging
from os.path import exists, join, isfile
from os import makedirs, listdir, unlink
from video_processor import VideoProcessor
from audio_transcriber import AudioTranscriber
from api import OllamaClient

def setup_environment(config):
    if (config['purge_cache']):
        # Remove all cached files
        for folder in ["transcriptions", "summaries", "audios"]:
            for file in listdir(folder):
                file_path = join(folder, file)
                try:
                    if isfile(file_path):
                        unlink(file_path)
                except Exception as e:
                    logging.error(f"Error deleting file {file_path}: {e}")
    else:
        # Create directories for transcriptions and summaries for caching
        makedirs("transcriptions", exist_ok=True)
        makedirs("summaries", exist_ok=True)
        makedirs("audios", exist_ok=True)

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

def summarize_with_llm(prompt, system_prompt, api: OllamaClient):
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
