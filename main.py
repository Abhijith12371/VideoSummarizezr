import os
from transformers import pipeline
import librosa
import soundfile as sf
import torch
from moviepy import VideoFileClip
import re
from collections import defaultdict

class VideoSummarizer:
    def __init__(self):
        # Optimized model selection
        self.asr_model = "openai/whisper-small"  # More accurate for tutorials
        self.summarization_model = "philschmid/bart-large-cnn-samsum"  # Better for conversational content
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize pipelines with better config
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.asr_model,
            device=self.device,
            chunk_length_s=30  # Better for Whisper
        )
        
        self.summarizer = pipeline(
            "summarization",
            model=self.summarization_model,
            device=self.device
        )

    def extract_audio(self, video_path):
        """Higher quality audio extraction"""
        try:
            video = VideoFileClip(video_path)
            audio_path = "temp_audio.wav"
            video.audio.write_audiofile(audio_path,
                                      codec='pcm_s16le',
                                      fps=16000,
                                      ffmpeg_params=["-ac", "1"])  # Force mono
            return audio_path
        except Exception as e:
            print(f"Audio extraction failed: {e}")
            return None

    def transcribe_audio(self, audio_path):
        """Enhanced transcription with correction"""
        try:
            # Whisper handles chunking internally
            result = self.asr_pipeline(audio_path)
            raw_text = result["text"]
            
            # Advanced correction system
            return self._correct_transcript(raw_text)
        except Exception as e:
            print(f"Transcription failed: {e}")
            return None

    def _correct_transcript(self, text):
        """Multi-layer correction system"""
        # Common error patterns in programming tutorials
        correction_patterns = {
            r"\bPITON\b": "Python",
            r"\bPYTHEN\b": "Python",
            r"\bPIE CHARM\b": "PyCharm",
            r"\bGUGAL\b": "Google",
            r"\bVERABLES\b": "variables",
            r"\bCOTE EDITOR\b": "code editor",
            r"\bSHONGYUHAIVIN\b": "showing you how to",
            r"\bCA SENSITIVE\b": "case sensitive",
            r"\bOPEN SAUCE\b": "open source",
            r"\b(\w+) (\w+) (?:OUT|EDIATELY)\b": r"\1 \2",  # Remove repeated phrases
            r"\b(\w+)\s+\1\b": r"\1"  # Remove duplicate words
        }
        
        # Apply corrections
        for pattern, replacement in correction_patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Fix punctuation and spacing
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = re.sub(r'([.,!?])(\w)', r'\1 \2', text)
        
        # Capitalize sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [sentence[0].upper() + sentence[1:] if sentence else "" 
                   for sentence in sentences]
        
        return ' '.join(sentences)

    def summarize_text(self, text):
        """Context-aware summarization"""
        if not text or len(text.split()) < 50:
            return text
            
        try:
            # Calculate lengths based on content
            word_count = len(text.split())
            max_len = min(130, max(50, word_count//4))
            min_len = min(30, max_len//2)
            
            # Context-aware summarization
            summary = self.summarizer(
                text,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                truncation=True
            )
            
            # Post-process summary
            return self._clean_summary(summary[0]["summary_text"])
        except Exception as e:
            print(f"Summarization failed: {e}")
            return self._fallback_summary(text)

    def _clean_summary(self, summary):
        """Clean summary output"""
        # Remove repeated phrases
        summary = re.sub(r'\b(\w+)\s+\1\b', r'\1', summary)
        # Fix capitalization
        return summary.capitalize()

    def _fallback_summary(self, text):
        """Simple extractive summary when abstractive fails"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        key_sentences = [s for s in sentences if len(s.split()) > 5]
        return ' '.join(key_sentences[:3])

    def summarize_video(self, video_path):
        """Complete processing pipeline with progress tracking"""
        print("\nStarting video processing...")
        
        print("\n1. Extracting audio...")
        audio_file = self.extract_audio(video_path)
        if not audio_file:
            return None
        
        print("\n2. Transcribing audio (this may take a while)...")
        transcription = self.transcribe_audio(audio_file)
        os.remove(audio_file)
        
        if not transcription:
            return None
            
        print("\n3. Generating summary...")
        summary = self.summarize_text(transcription)
        
        return {
            "transcription": transcription,
            "summary": summary
        }

if __name__ == "__main__":
    summarizer = VideoSummarizer()
    video_path = input("Enter path to local video file: ").strip()
    
    if not os.path.exists(video_path):
        print("Error: File not found!")
        exit()
    
    result = summarizer.summarize_video(video_path)
    
    if result:
        print("\n=== Cleaned Transcription ===")
        print(result["transcription"][:1000] + ("..." if len(result["transcription"]) > 1000 else ""))
        
        print("\n=== Final Summary ===")
        print(result["summary"])
        
        # Save results with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with open(f"transcription_{timestamp}.txt", "w") as f:
            f.write(result["transcription"])
        
        with open(f"summary_{timestamp}.txt", "w") as f:
            f.write(result["summary"])
    else:
        print("Processing failed")