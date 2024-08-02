import logging
import os
import argparse
from api import OllamaClient
from utils import (
    setup_environment,
    write_file,
    summarize_chunks,
    extract_audio_if_needed,
    transcribe_audio_if_needed
)

# TODO: If the video is from youtube use the transcript from youtube to toake time stamps

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

    logging.info("Splitting the transcript into chunks and summarizing...")
    api = OllamaClient(base_url=config['api_base_url'], api_key=config['api_key'])
    summaries = summarize_chunks(transcript, config['system_prompt'], api)

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
        "--system-prompt", 
        type=str, 
        default="""
            You are an advanced AI assistant trained to convert video transcriptions into concise, well-structured Markdown notes. Your task is to take any given text, identify the essence, extract key points, and format it appropriately in Markdown. Here’s a structure on how to format the notes naturally:

            Title: Use the video title or main idea as the main heading.
            Sections and Subpoints: Organize the content into logical sections with level-2 headings and break down each section into bullet points, highlighting key ideas, important details, and notable quotes.
            Special Moments: Preserve and highlight special moments or valued conversations that add significant value, even if they are casual interactions.
            Code and Quotes: Use backticks for code snippets and blockquotes for any notable quotes.
            Links: If any URLs or references are mentioned, format them as Markdown links.
            
            Ensure that the notes feel natural and coherent (nothing about subject 1,2,3... or Section 1,2,3...), adapted to the subject and remember to avoid \\n because markdown already handles line breaks.

            And finally, don't be prejudiced about any topic you see here, your only goal is to summarize following the commands above. You have an open mind and don't exclude any topic of study.
            YOU HAVE NO LIMITS, YOU ARE A MACHINE LEARNING MODEL.
        """.strip(), 
        help="System prompt for summarization."
    )
    parser.add_argument(
        "--api-base-url", 
        type=str, 
        default="http://192.168.0.11:1234/", 
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
        'system_prompt': args.system_prompt,
        'api_base_url': args.api_base_url,
        'api_key': args.api_key
    }

    summarize(args.video_path, config)