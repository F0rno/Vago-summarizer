import logging
import os
import argparse
from transformers import AutoTokenizer
from api import OpenAIClient
from utils import (
    setup_environment,
    write_file,
    split_transcript,
    summarize_chunks,
    extract_audio_if_needed,
    transcribe_audio_if_needed
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def summarize(video_path, config):
    setup_environment()
    
    # Create directories for transcriptions and summaries if they don't exist
    os.makedirs("transcriptions", exist_ok=True)
    os.makedirs("summaries", exist_ok=True)
    
    audio_path = video_path.replace("videos/", "audios/").replace(".mp4", ".mp3")
    file_name = video_path.split("/")[-1].replace(".mp4", "")
    transcript_path = f"transcriptions/{file_name}-transcript.txt"
    summary_path = f"summaries/{file_name}-summary.md"

    extract_audio_if_needed(video_path, audio_path)
    transcript = transcribe_audio_if_needed(audio_path, transcript_path)

    logging.info("Loading the model...")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    prompt_length = len(tokenizer.tokenize(config['system_prompt']))
    max_chunk_length = config['context_window'] - prompt_length

    logging.info("Splitting the transcript into chunks and summarizing...")
    chunks = split_transcript(transcript, max_chunk_length)
    api = OpenAIClient(base_url=config['api_base_url'], api_key=config['api_key'])
    summaries = summarize_chunks(chunks, config['system_prompt'], api)

    logging.info("Writing the summaries to a file...")
    write_file(summary_path, "\n".join(summaries))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video and audio files.")
    parser.add_argument(
        "--video-path",
        # TODO: Remove this
        default="videos/index.mp4",
        type=str, 
        help="Path to the video file."
    )
    parser.add_argument(
        "--context-window", 
        type=int, 
        default=4000, 
        help="Context window size for processing."
    )
    parser.add_argument(
        "--system-prompt", 
        type=str, 
        default="""Forma this transcript into a summary of the video using markdown for notes. Only prompt the formated text, nothing more:\n\n""", 
        help="System prompt for summarization."
    )
    parser.add_argument(
        "--api-base-url", 
        type=str, 
        default="http://192.168.0.11:1234/v1", 
        help="Base URL for the API."
    )
    parser.add_argument(
        "--api-key", 
        type=str, 
        default="lm-studio", 
        help="API key for authentication."
    )
    
    args = parser.parse_args()

    config = {
        'context_window': args.context_window,
        'system_prompt': args.system_prompt,
        'api_base_url': args.api_base_url,
        'api_key': args.api_key
    }

    summarize(args.video_path, config)