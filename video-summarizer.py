from transformers import pipeline, AutoTokenizer
from video_processor import VideoProcessor
from audio_transcriber import AudioTranscriber
from os.path import exists
from api import OpenAIClient
from os import environ

environ["TOKENIZERS_PARALLELISM"] = "false"

def file_exists(file_path):
    return exists(file_path)

def split_transcript(transcript, max_length):
    return [transcript[i:i+max_length] for i in range(0, len(transcript), max_length)]

def summarize_chunks(chunks, system_prompt, api: OpenAIClient):
    list = []
    for chunk in chunks:
        list.append(api.call_llm(chunk, system_prompt))
    return list

def summarize(video_path):
    audio_path = video_path.replace("videos/", "audios/").replace(".mp4", ".mp3")
    file_name = video_path.split("/")[-1].replace(".mp4", "")
    if not file_exists(audio_path):
        # Creates a file with the audio extracted from the video
        VideoProcessor(video_path, audio_path).extract_audio()
    transcript = ""
    if not file_exists(f"{file_name}-transcript.txt"):
        transcript = AudioTranscriber(audio_path).transcribe()
        # Save the transcript to a file
        with open(f"{file_name}-transcript.txt", "w") as transcript_file:
            transcript_file.write(transcript)
    else:
        with open(f"{file_name}-transcript.txt", "r") as transcript_file:
            transcript = transcript_file.read()
    print("Loading the model...")
    system_prompt = """
    Forma this transcript into a summary of the video using markdown for notes. Only prompt the formated text, nothing more:\n\n
    """.strip()
    api = OpenAIClient(base_url="http://192.168.0.11:1234/v1", api_key="lm-studio")
    print("Splitting the transcript into chunks and summarizing...")

    # Calculate the length of the system prompt in tokens
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    prompt_length = len(tokenizer.tokenize(system_prompt))

    # Subtract the prompt length from the context window
    context_window = 4000
    max_chunk_length = context_window - prompt_length

    chunks = split_transcript(transcript, max_chunk_length)
    print("Summarizing...")
    summaries = summarize_chunks(chunks, system_prompt, api)
    
    print("Writing the summaries to a file...")
    with open(f"{file_name}-summary.md", "w") as summary_file:
        summary_file.write("\n".join(summaries))

if __name__ == "__main__":
    video_path = "videos/index.mp4"
    summarize(video_path)