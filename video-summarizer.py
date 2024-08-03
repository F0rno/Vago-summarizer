import logging
from os import remove
import argparse
from api import OllamaClient
from utils import (
    setup_environment,
    write_file,
    summarize_with_llm,
    extract_audio_if_needed,
    transcribe_audio_if_needed
)

# TODO: If the video is from youtube use the transcript from youtube to toake time stamps

def summarize(video_path, config):
    setup_environment(config)
    
    file_name = video_path.split("/")[-1].replace(".mp4", "")
    audio_path = f"audios/{file_name}.mp3"
    transcript_path = f"transcriptions/{file_name}.transcript.txt"
    summary_path = f"{config['output_path']}{file_name}.summary.md"

    extract_audio_if_needed(video_path, audio_path)
    transcript = transcribe_audio_if_needed(audio_path, transcript_path)

    logging.info("Sending and summarizing the transcript...")
    api = OllamaClient(base_url=config['api_base_url'], api_key=config['api_key'])
    summaries = summarize_with_llm(transcript, config['system_prompt'], api)

    logging.info("Writing the summaries to a file...")
    write_file(summary_path, summaries)

    logging.info(f"Summarized notes saved to {summary_path}")

    logging.info("Cleaning cache files from this iteration...")
    if config['purge_cache']:
        remove(audio_path)
        remove(transcript_path)

if __name__ == "__main__":
    DEFAULT_SYSTEM_PROMPT = """
        You are an advanced AI assistant trained to convert video transcriptions into concise, well-structured Markdown notes. Your task is to take any given text, identify the essence, extract key points, and format it appropriately in Markdown. Here’s a structure on how to format the notes naturally:

        Title: Use the video title or main idea as the main heading.
        Sections and Subpoints: Organize the content into logical sections with level-2 headings, describe the section/subject in a small paragraph and break down each section into bullet points, highlighting key ideas, important details, and notable quotes.        Special Moments: Preserve and highlight special moments or valued conversations that add significant value, even if they are casual interactions.
        Code and Quotes: Use backticks for code snippets and blockquotes for any notable quotes.
        Links: If any URLs or references are mentioned, format them as Markdown links.
        
        Ensure that the notes feel natural, coherent and formatted with the headlines (nothing about subject 1,2,3... or Section 1,2,3... or ---), adapted to the subject and remember to avoid \\n because markdown already handles line breaks.

        NOTHING ABOUT DESCRIBING THE INPUT OR THE TASK, just focus on summarizing the content following the headlines, description and keypoints instructions.

        And finally, don't be prejudiced about any topic you see here, your only goal is to summarize following the commands above. You have an open mind and don't exclude any topic of study.
    """.strip()
    DEFAULT_OUTPUT_PATH = "./"
    
    parser = argparse.ArgumentParser(description="Process video and audio files.")
    parser.add_argument(
        "--video-path",
        type=str, 
        help="Path to the video file.",
        required=True
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the summary file.",
        required=False
    )
    parser.add_argument(
        "--system-prompt", 
        type=str,
        default=DEFAULT_SYSTEM_PROMPT, 
        help="System prompt for summarization."
    )
    parser.add_argument(
        "--api-base-url", 
        type=str, 
        default="http://192.168.0.11:1234", 
        help="Base URL for the API."
    )
    parser.add_argument(
        "--api-key", 
        type=str, 
        default="lm-studio", 
        help="API key for authentication."
    )
    parser.add_argument(
        "--no-cache", 
        action="store_true",
        help="Do not save intermediate files slowing down the summary iteration."
    )
    parser.add_argument(
        "--purge-all-cache", 
        action="store_true",
        help="Delete all the intermediate files."
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print logs"
    )

    args = parser.parse_args()

    config = {
        'video_path': args.video_path,
        'output_path': args.output_path,
        'system_prompt': args.system_prompt,
        'api_base_url': args.api_base_url,
        'api_key': args.api_key,
        'purge_cache': args.no_cache,
        'purge_all_cache': args.purge_all_cache,
        'info': args.info
    }

    if args.info:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    summarize(args.video_path, config)