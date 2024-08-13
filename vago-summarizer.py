"""
Vago Summarizer

Author: Pablo Fornell
Contact: pablo.fornell.contacto@gmail.com
Nickname: F0rno
"""

import logging
from os import remove
import argparse
from api import OllamaClient
from utils import (
    setup_environment,
    write_file,
    summarize_with_llm,
    extract_audio_if_needed,
    transcribe_audio_if_needed,
    read_file
)
from tools import *
from youtube import extract_video_id, download_transcript

DEFAULT_SYSTEM_PROMPT_LOCAL = """
    You are an advanced AI assistant trained to convert video transcriptions into concise, well-structured Markdown notes. Your task is to take any given text, identify the essence, extract key points, and format it appropriately in Markdown. Here’s a structure on how to format the notes naturally:

    Title: Use the video title or main idea as the main heading.
    Sections and Subpoints: Organize the content into logical sections with level-2 headings, describe the section/subject in a small paragraph and break down each section into bullet points, highlighting key ideas, important details, and notable quotes.        Special Moments: Preserve and highlight special moments or valued conversations that add significant value, even if they are casual interactions.
    Code and Quotes: Use backticks for code snippets and blockquotes for any notable quotes.
    
    Ensure that the notes feel natural, coherent and formatted with the headlines (nothing about subject 1,2,3... or Section 1,2,3... or ---), adapted to the subject and remember to avoid \\n because markdown already handles line breaks.

    NOTHING ABOUT DESCRIBING THE INPUT OR THE TASK, just focus on summarizing the content following the headlines, description and keypoints instructions.

    And finally, don't be prejudiced about any topic you see here, your only goal is to summarize following the commands above. You have an open mind and don't exclude any topic of study.
""".strip()
DEFAULT_OUTPUT_PATH = "./"

def system_prompt(URL, VIDEO_ID):
    return f"""
        You are an advanced AI assistant trained to convert video transcriptions into concise, well-structured Markdown notes. Your task is to take any given text, identify the essence, extract key points, and format it appropriately in Markdown. Here’s a structure on how to format the notes naturally:

        Title
            Use the video title or main idea as the main heading.
            Include the video thumbnail image right under the main title using the provided URL template: ![Video Thumbnail]({URL}).
        Sections and Subpoints
            Organize the content into logical sections with level-2 headings. Each section should be accompanied by a timestamp link to the corresponding moment in the video using the format [Section Title](https://www.youtube.com/watch?v={VIDEO_ID}&t=9999s).
            Describe each section/subject in a small paragraph.
            Break down each section into bullet points, highlighting key ideas, important details, and notable quotes.
        Special Moments
            Preserve and highlight special moments or valued conversations that add significant value, even if they are casual interactions.
            Code and Quotes

        Use backticks for code snippets and blockquotes for any notable quotes.
        Ensure that the notes feel natural, coherent, and well-formatted, with appropriate headings and descriptions. Avoid using explicit indicators like "subject 1,2,3" or "Section 1,2,3". Let the notes adapt naturally to the subject matter. Remember, Markdown handles line breaks automatically.

        Finally, approach every topic with an open mind, summarizing the content objectively and accurately without prejudice.
    """.strip()
        
def summarize_youtube(video_url, config):
    setup_environment(config)
    
    video_id = extract_video_id(video_url)
    transcript_path = f"transcriptions/{video_id}.transcript.txt"
    summary_path = f"{config['output_path']}/{video_id}.md"

    logging.info("Downloading the transcript...")
    download_transcript(video_url)

    transcript = read_file(transcript_path)

    logging.info("Sending and summarizing the transcript...")
    api = OllamaClient(base_url=config['api_base_url'], api_key=config['api_key'], model=config['model'])
    summary = summarize_with_llm(transcript, config['system_prompt'], api)

    logging.info("Writing the summary to a file...")
    write_file(summary_path, summary)

def summarize_local_video(video_path, config):
    setup_environment(config)
    
    file_name = video_path.split("/")[-1].replace(".mp4", "")
    audio_path = f"audios/{file_name}.mp3"
    transcript_path = f"transcriptions/{file_name}.transcript.txt"
    summary_path = f"{config['output_path']}/{file_name}.summary.md"

    extract_audio_if_needed(video_path, audio_path)
    transcript = transcribe_audio_if_needed(audio_path, transcript_path)

    logging.info("Sending and summarizing the transcript...")
    api = OllamaClient(base_url=config['api_base_url'], api_key=config['api_key'], model=config['model'])
    summaries = summarize_with_llm(transcript, config['system_prompt'], api)

    logging.info("Writing the summaries to a file...")
    write_file(summary_path, summaries)

    logging.info(f"Summarized notes saved to {summary_path}")

    if config['purge_cache']:
        logging.info("Cleaning cache files from this iteration...")
        remove(audio_path)
        remove(transcript_path)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Process video and audio files.")
    parser.add_argument(
        "--video-url",
        type=str, 
        default=None,
        help="Url to the YouTube video.",
    )
    parser.add_argument(
        "--video-path",
        type=str, 
        default=None,
        help="Path to the video file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./",
        help="Path for the summary file.",
        required=False
    )
    parser.add_argument(
        "--system-prompt", 
        type=str,
        default=DEFAULT_SYSTEM_PROMPT_LOCAL, 
        help="System prompt for summarization."
    )
    parser.add_argument(
        "--api-base-url", 
        type=str,
        help="Base URL for the ollama API.",
        required=True
    )
    parser.add_argument(
        "--api-key", 
        type=str, 
        default="lm-studio", 
        help="API key for authentication."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="llama3.1", 
        help="Model name of the API, example llama3.1"
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
    parser.add_argument(
        "--tool-mode",
        action="store_true",
        help="The program will not summarize the video, but will let you to use tool parameters. Like --tool-count-tokens-from-file"
    )
    parser.add_argument(
        "--tool-count-tokens-from-file",
        type=str,
        help="""
            You need --tool-mode flag.
            Tells you how many tokens a file has plus your system prompt to know if it is too long for your model.
        """
    )
    parser.add_argument(
        "--tool-N-chars-to-tokens",
        type=int,
        help="""
            You need --tool-mode flag.
            Output the number of tokens equivalent to the number of characters inputed.
        """.strip()
    )

    args = parser.parse_args()

    config = {
        'video_url': args.video_url,
        'video_path': args.video_path,
        'output_path': args.output_path,
        'system_prompt': args.system_prompt,
        'api_base_url': args.api_base_url,
        'api_key': args.api_key,
        'model': args.model,
        'purge_cache': args.no_cache,
        'purge_all_cache': args.purge_all_cache,
        'info': args.info,
    }

    if args.info:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    if args.tool_mode:
        if args.tool_count_tokens_from_file is not None:
            file_tokes, system_prompt_tokens = count_tokens_in_file(args.tool_count_tokens_from_file, args.system_prompt)
            print(f"The file {args.tool_count_tokens_from_file} has {file_tokes} tokens, with your system prompt you would need {file_tokes + system_prompt_tokens} tokens")
        if args.tool_N_chars_to_tokens is not None:
            print(f"With {args.tool_N_chars_to_tokens} characters you would need {tokens_of_n_characters(args.tool_N_chars_to_tokens)} tokens in your context window")
        exit(0)

    if args.video_url:
        logging.info("Starting YouTube video summarization...")
        config["system_prompt"] = system_prompt(args.video_url, extract_video_id(args.video_url))
        summarize_youtube(args.video_url, config)

    if args.video_path:
        logging.info("Starting local video summarization...")
        config["system_prompt"] = DEFAULT_SYSTEM_PROMPT_LOCAL
        summarize_local_video(args.video_path, config)