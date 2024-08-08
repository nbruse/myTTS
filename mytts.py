"""
Text-to-Speech Conversion Script

This script converts text from various file formats (PDF, EPUB, TXT) into speech using TTS models.
It supports both English and German languages, processes text in chunks, and can combine audio files.

Usage:
    python script_name.py <file_path> [options]

Arguments:
    file_path             Path to the input file (PDF, EPUB, or TXT)

Options:
    --output-dir          Directory to save audio files (default: "output_audio")
    --start-chunk         Starting chunk number for processing (default: 0)
    --max-chunks          Maximum number of chunks to process
    --no-sentence-splitting   Disable sentence splitting technique
    --combine-audio       Combine audio files: 0 for no combining, 1 for all in one, >1 for grouping

Requirements:
    - TTS
    - PyPDF2
    - ebooklib
    - beautifulsoup4
    - langdetect
    - pydub
    - filetype

Note: This script requires significant computational resources and may take a while to process large files.
"""

import os
import gc
import multiprocessing
from TTS.api import TTS
import argparse
import filetype
import PyPDF2
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re
from langdetect import detect, LangDetectException
from pydub import AudioSegment

def detect_file_format(file_path):
    """
    Detect the format of the input file.
    
    Args:
        file_path (str): Path to the input file.
    
    Returns:
        str: Detected file format ('pdf', 'epub', 'text', or 'unknown').
    """
    kind = filetype.guess(file_path)
    if kind is None:
        # If filetype can't determine the type, check if it's a text file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read()
            return 'text'
        except UnicodeDecodeError:
            return 'unknown'
    
    mime = kind.mime
    extension = os.path.splitext(file_path)[1].lower()
    
    if mime == 'application/pdf' or extension == '.pdf':
        return 'pdf'
    elif mime == 'application/epub+zip' or extension == '.epub':
        return 'epub'
    elif mime.startswith('text/') or extension in ['.txt', '.md']:
        return 'text'
    else:
        return 'unknown'

def extract_text(file_path):
    """
    Extract text content from the input file.
    
    Args:
        file_path (str): Path to the input file.
    
    Returns:
        str: Extracted text content.
    
    Raises:
        ValueError: If the file format is unsupported.
    """
    format = detect_file_format(file_path)
    if format == 'pdf':
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = "".join(page.extract_text() for page in reader.pages)
    elif format == 'epub':
        book = epub.read_epub(file_path)
        text = "\n".join(BeautifulSoup(item.get_content(), 'html.parser').get_text()
                         for item in book.get_items() if item.get_type() == ebooklib.ITEM_DOCUMENT)
    elif format == 'text':
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    else:
        raise ValueError(f"Unsupported file format: {format}")
    
    return re.sub(r'\s+', ' ', text).strip()

def split_long_sentence(sentence, max_length=100):
    """
    Split a long sentence into smaller parts.
    
    Args:
        sentence (str): The sentence to split.
        max_length (int): Maximum length of each part.
    
    Returns:
        list: List of sentence parts.
    """
    if len(sentence) <= max_length:
        return [sentence]
    
    split_points = [m.start() for m in re.finditer(r'[,;]|\s(and|but|or|und|aber|oder)\s', sentence)]
    
    if not split_points:
        return [sentence[:max_length], sentence[max_length:]]
    
    best_split = min(split_points, key=lambda x: abs(x - len(sentence)//2))
    return [sentence[:best_split+1].strip(), sentence[best_split+1:].strip()]

def process_chunk(chunk, output_file, language, use_sentence_splitting=True):
    """
    Process a text chunk and convert it to speech.
    
    Args:
        chunk (str): Text chunk to process.
        output_file (str): Path to save the output audio file.
        language (str): Language of the text ('en' or 'de').
        use_sentence_splitting (bool): Whether to use sentence splitting.
    """
    if language == 'de':
        tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC")
        speaker = None
    elif language == 'en':
        tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False)
        speaker = "p234"
    else:
        raise ValueError(f"Unsupported language: {language}")

    try:
        if use_sentence_splitting:
            sentences = re.split(r'(?<=[.!?])\s+', chunk)
            audio_segments = []
            for sentence in sentences:
                parts = split_long_sentence(sentence)
                for part in parts:
                    temp_file = f"{output_file}_temp.wav"
                    tts.tts_to_file(text=part, file_path=temp_file, speaker=speaker)
                    audio_segments.append(AudioSegment.from_wav(temp_file))
                    os.remove(temp_file)
            combined = sum(audio_segments)
            combined.export(output_file, format="wav")
        else:
            tts.tts_to_file(text=chunk, file_path=output_file, speaker=speaker)
        
        print(f"Successfully processed chunk and saved to {output_file}")
    except Exception as e:
        print(f"Error processing chunk: {e}")
        print("Chunk content:", chunk)

    del tts
    gc.collect()

def chunk_generator(text, chunk_size=4000):
    """
    Generate chunks of text.
    
    Args:
        text (str): Input text to chunk.
        chunk_size (int): Maximum size of each chunk.
    
    Yields:
        str: Text chunks.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current_chunk = []
    current_size = 0
    for sentence in sentences:
        if current_size + len(sentence) <= chunk_size:
            current_chunk.append(sentence)
            current_size += len(sentence) + 1
        else:
            if current_chunk:
                yield ' '.join(current_chunk)
            current_chunk = [sentence]
            current_size = len(sentence)
    if current_chunk:
        yield ' '.join(current_chunk)

def combine_audio_files(input_files, output_file, delete_input=True):
    """
    Combine multiple audio files into one.
    
    Args:
        input_files (list): List of input audio file paths.
        output_file (str): Path to save the combined audio file.
        delete_input (bool): Whether to delete input files after combining.
    """
    combined = AudioSegment.empty()
    for file in input_files:
        audio = AudioSegment.from_wav(file)
        combined += audio
        del audio
        gc.collect()
        if delete_input:
            os.remove(file)
    combined.export(output_file, format="wav")
    del combined
    gc.collect()

def text_to_speech(text, output_dir, language, start_chunk=0, max_chunks=None, use_sentence_splitting=True, combine_audio=0):
    """
    Convert text to speech and save as audio files.
    
    Args:
        text (str): Input text to convert.
        output_dir (str): Directory to save output audio files.
        language (str): Language of the text ('en' or 'de').
        start_chunk (int): Starting chunk number for processing.
        max_chunks (int): Maximum number of chunks to process.
        use_sentence_splitting (bool): Whether to use sentence splitting.
        combine_audio (int): How to combine audio files (0: no combining, 1: all in one, >1: grouping).
    """
    os.makedirs(output_dir, exist_ok=True)

    chunks = list(chunk_generator(text))
    end_chunk = min(start_chunk + (max_chunks or len(chunks)), len(chunks))
    chunks_to_process = chunks[start_chunk:end_chunk]
    
    print(f"Processing {len(chunks_to_process)} chunks (from {start_chunk + 1} to {end_chunk})")
    
    output_files = []
    for i, chunk in enumerate(chunks_to_process, start=start_chunk+1):
        output_file = os.path.join(output_dir, f"audio_chunk_{i}.wav")
        
        p = multiprocessing.Process(target=process_chunk, args=(chunk, output_file, language, use_sentence_splitting))
        p.start()
        p.join()
        
        output_files.append(output_file)
        print(f"Processed chunk {i}/{end_chunk}")
        gc.collect()

    print("Audio generation complete")

    if combine_audio > 0:
        if combine_audio == 1:
            # Combine all chunks into one file
            combined_file = os.path.join(output_dir, "combined_audio.wav")
            combine_audio_files(output_files, combined_file)
            print(f"All audio chunks combined into {combined_file}")
        else:
            # Combine chunks into groups
            for i in range(0, len(output_files), combine_audio):
                group = output_files[i:i+combine_audio]
                combined_file = os.path.join(output_dir, f"combined_audio_group_{i//combine_audio+1}.wav")
                combine_audio_files(group, combined_file, delete_input=False)
                print(f"Audio chunks {i+1} to {i+len(group)} combined into {combined_file}")
        
        # Optionally delete individual chunk files after combining
        if combine_audio == 1:
            for file in output_files:
                os.remove(file)
            print("Individual chunk files deleted after combining.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a book file for TTS conversion")
    parser.add_argument("file_path", help="Path to the book file (PDF or EPUB)")
    parser.add_argument("--output-dir", default="output_audio", help="Directory to save audio files")
    parser.add_argument("--start-chunk", type=int, default=0, help="Starting chunk number")
    parser.add_argument("--max-chunks", type=int, help="Maximum number of chunks to process")
    parser.add_argument("--no-sentence-splitting", action="store_true", help="Disable sentence splitting technique")
    parser.add_argument("--combine-audio", type=int, default=0, 
                        help="Combine audio files: 0 for no combining, 1 for all in one, >1 for grouping")
    
    args = parser.parse_args()
    
    try:
        text = extract_text(args.file_path)
        
        language = detect(text[:1000])
        if language not in ['de', 'en']:
            print(f"Detected language {language} is not supported. Defaulting to English.")
            language = 'en'
        
        print(f"Detected language: {language}")
        
        text_to_speech(text, args.output_dir, language, start_chunk=args.start_chunk, 
                       max_chunks=args.max_chunks, use_sentence_splitting=not args.no_sentence_splitting,
                       combine_audio=args.combine_audio)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()