import logging
import argparse
from os.path import exists
from os import environ
from video_processor import VideoProcessor
from audio_transcriber import AudioTranscriber
from api import OpenAIClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DEFAULT_CONTEXT_WINDOW = 4000
DEFAULT_SYSTEM_PROMPT = """
Forma this transcript into a summary of the video using markdown for notes. Only prompt the formated text, nothing more:\n\n
""".strip()
DEFAULT_API_BASE_URL = "http://192.168.0.11:1234/v1"
DEFAULT_API_KEY = "lm-studio"

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

def split_transcript(transcript, max_length):
    # TODO: This spliter is bad implemented because max_length is the len of tokes
    # not normal text
    # TODO: Add a parameter to determine the lm and select an acurate tokenizer
    return [transcript[i:i+max_length] for i in range(0, len(transcript), max_length)]

def summarize_chunks(chunks, system_prompt, api: OpenAIClient):
    return [api.call_llm(chunk, system_prompt) for chunk in chunks]

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
