from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import logging

def extract_video_id(url):
    query = urlparse(url).query
    params = parse_qs(query)
    return params.get('v', [None])[0]

def download_transcript(video_url):
    video_id = extract_video_id(video_url)
    if not video_id:
        logging.info("Invalid YouTube URL")
        exit(1)
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'es'])
        with open(f"transcriptions/{video_id}.transcript.txt", "w") as file:
            for entry in transcript:
                start = entry['start']
                duration = entry['duration']
                text = entry['text']
                file.write(f"{start:.2f}-{start+duration:.2f} {text}\n")
        logging.info(f"Transcript for video {video_id} has been downloaded successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    video_url = ''
    download_transcript(video_url)