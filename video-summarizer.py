from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from video_processor import VideoProcessor
from audio_transcriber import AudioTranscriber


def split_transcript(transcript, max_length=1024):
    return [transcript[i:i+max_length] for i in range(0, len(transcript), max_length)]

def summarize_chunks(chunks, summarizer):
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return summaries

def summarize(video_path):
    audio_path = video_path.replace("videos/", "audios/").replace(".mp4", ".mp3")
    audio_transcriber = AudioTranscriber(audio_path)
    if not audio_transcriber.check_if_audio_exists():
        VideoProcessor(video_path, audio_path).extract_audio()
    transcript = AudioTranscriber(audio_path).transcribe()

    # Save the transcript to a file
    with open(f"{file_name}-transcript.txt", "w") as transcript_file:
        transcript_file.write(transcript)

    ############################################################
    # Summarization model
    ############################################################
    print("Loading te model...")
    model_name = "facebook/bart-large-cnn"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=0)

    print("Splitting the transcript into chunks and summarizing...")
    chunks = split_transcript(transcript)
    print("Summarizing...")
    summaries = summarize_chunks(chunks, summarizer)
    
    print("Writing the summaries to a file...")
    with open(f"{file_name}-summary.md", "w") as summary_file:
        summary_file.write("\n".join(summaries))

if __name__ == "__main__":
    video_path = "videos/index.mp4"
    file_name = video_path.split("/")[-1].replace(".mp4", "")
    summarize(video_path)