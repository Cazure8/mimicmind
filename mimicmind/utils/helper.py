# The MIT License (MIT)
# Copyright © 2025 Cazure

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import torch
import re
import random
import requests

from io import BytesIO
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()

proxy_url = os.getenv('PROXY_URL')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
# Set pad token
tokenizer.pad_token = tokenizer.eos_token

def transcribe_with_whisper(audio_filepath):
    """Transcribe audio using Whisper.
    Used for evaluating the accuracy of cloned voices."""
    import whisper
    model = whisper.load_model("base")
    result = model.transcribe(audio_filepath)
    return result["text"]

def fetch_random_sentences(context="MimicMind is the revolution of voice cloning.", num_sentences=3):
    """Generate random sentences for voice cloning tasks."""
    # Tokenize input context
    input_ids = tokenizer.encode(context, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)  # Create attention mask

    # Generate text
    outputs = model.generate(
        input_ids,
        max_length=100,  # Ensure sufficient length for multiple sentences
        num_return_sequences=1,  # Generate a single long sequence
        do_sample=True,
        top_p=0.9,  # Adjust for slightly less randomness
        top_k=50,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and clean text
    decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if context in decoded_text:
        decoded_text = decoded_text.replace(context, "")
        
    # Split the text into sentences using regular expressions
    split_sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', decoded_text)
    cleaned_sentences = [
        re.sub(r"[^a-zA-Z0-9.,'\" ]", "", s.strip())  # Remove unwanted characters
        for s in split_sentences
        if len(s.strip()) > 0
    ]

    # Concatenate the required number of sentences
    if len(cleaned_sentences) >= num_sentences:
        return " ".join(cleaned_sentences[:num_sentences])
    else:
        return " ".join(cleaned_sentences)  # Return all sentences if fewer are available
    
def get_random_audio_clip(directory="datasets"):
    """Get a random audio clip from the specified directory for voice cloning reference."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} not found. Please create it and add audio files.")
        
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter for audio files
    audio_files = [f for f in files if f.endswith((".mp3", ".wav"))]
    
    # Check if there are any audio files
    if not audio_files:
        raise ValueError("No audio files found in the directory. Please add .mp3 or .wav files.")
    
    # Select a random file
    random_file = random.choice(audio_files)
    
    # Return the full path of the random file
    return os.path.join(directory, random_file)

def handle_filename_duplicates(filepath):
    """Ensure the filepath is unique to avoid overwriting existing files."""
    base, extension = os.path.splitext(filepath)
    counter = 1
    while os.path.exists(filepath):
        filepath = f"{base}_{counter}{extension}"
        counter += 1
    return filepath