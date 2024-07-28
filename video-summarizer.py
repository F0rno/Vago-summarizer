from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from video_processor import VideoProcessor
from audio_transcriber import AudioTranscriber
from datasets import Dataset
from os.path import exists
import torch

def file_exists(file_path):
    return exists(file_path)

def split_transcript(transcript, max_length=1024):
    return [transcript[i:i+max_length] for i in range(0, len(transcript), max_length)]

def summarize_chunks(chunks, summarizer):
    dataset = Dataset.from_dict({"text": chunks})
    summaries = summarizer(dataset["text"], max_length=130, min_length=30, do_sample=False, batch_size=2)  # Reduced batch size
    return [summary['summary_text'] for summary in summaries]

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
    ############################################################
    # Summarization model
    ############################################################
    print("Loading the model...")
    model_name = "facebook/bart-large-cnn"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    try:
        print("Using GPU...")
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=0)  # Use GPU
    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory. Switching to CPU...")
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)  # Use CPU

    print("Splitting the transcript into chunks and summarizing...")
    chunks = split_transcript(transcript)
    print("Summarizing...")
    summaries = summarize_chunks(chunks, summarizer)
    
    print("Writing the summaries to a file...")
    with open(f"{file_name}-summary.md", "w") as summary_file:
        summary_file.write("\n".join(summaries))

if __name__ == "__main__":
    video_path = "videos/adrian.mp4"
    summarize(video_path)