from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from video_processor import VideoProcessor
from audio_transcriber import AudioTranscriber


def split_transcript(transcript, max_length=1024):
    return [transcript[i:i+max_length] for i in range(0, len(transcript), max_length)]

def summarize_chunks(chunks):
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return summaries

if __name__ == "__main__":
    video_path = "index.mp4"
    video_processor = VideoProcessor(video_path)
    audio_path = video_processor.extract_audio()
    
    audio_transcriber = AudioTranscriber(audio_path)
    transcript = audio_transcriber.transcribe()
    
    # Save the transcript to a file
    with open("transcript.txt", "w") as transcript_file:
        transcript_file.write(transcript)

    ############################################################
    # Summarization model
    ############################################################
    model_name = "sshleifer/distilbart-cnn-12-6"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=0)

    chunks = split_transcript(transcript)
    summaries = summarize_chunks(chunks)
    
    # Save the summarization to a file
    with open("summary.md", "w") as summary_file:
        summary_file.write("\n".join(summaries))