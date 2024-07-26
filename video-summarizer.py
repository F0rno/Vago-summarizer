import subprocess
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from video_processor import VideoProcessor
from audio_transcriber import AudioTranscriber

model_name = "sshleifer/distilbart-cnn-12-6"

def load_model(model_name):
    try:
        # Try to load the cached model and tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except EnvironmentError:
        # If loading fails, download the model
        subprocess.run(["python3", "download_model.py"])
        # Retry loading the model and tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def split_transcript(transcript, max_length=1024):
    return [transcript[i:i+max_length] for i in range(0, len(transcript), max_length)]

def summarize_chunks(chunks, base_prompt):
    summaries = []
    for chunk in chunks:
        # Fill in the template with the current chunk
        prompt = base_prompt.format(chunk)
        summary = summarizer(prompt, max_length=130, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return summaries

# Load the model and tokenizer
model, tokenizer = load_model(model_name)

# Create the summarizer pipeline using the cached model and tokenizer
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=0)

prompt_template = "Summarize the following text in Markdown format:\n\n{}"

if __name__ == "__main__":
    video_path = "index.mp4"
    video_processor = VideoProcessor(video_path)
    audio_path = video_processor.extract_audio()
    
    audio_transcriber = AudioTranscriber(audio_path)
    transcript = audio_transcriber.transcribe()
    
    # Save the transcript to a file
    with open("transcript.txt", "w") as transcript_file:
        transcript_file.write(transcript)
    
    chunks = split_transcript(transcript)
    summaries = summarize_chunks(chunks, prompt_template)
    
    final_summary = " ".join(summaries)
    
    # Save the summarization to a file
    with open("summary.md", "w") as summary_file:
        summary_file.write(final_summary)