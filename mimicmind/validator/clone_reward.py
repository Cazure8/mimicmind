import os
import time
import base64
import torchaudio
import numpy as np
import librosa
from pydub import AudioSegment
from typing import List
from scipy.spatial.distance import cosine, euclidean
from speechbrain.inference import SpeakerRecognition
import whisper

# Global variables for preloaded models
VERIFICATION_MODEL = None
MOSNET_MODEL = None

# Directory to save cloned audio files
CLONED_AUDIO_DIR = "miner_cloned_voices"
os.makedirs(CLONED_AUDIO_DIR, exist_ok=True)  # Ensure the directory exists

# ---------------------------
# Model Loading Functions
# ---------------------------

def get_verification_model():
    """Load the SpeakerRecognition model once and reuse it."""
    global VERIFICATION_MODEL
    if VERIFICATION_MODEL is None:
        VERIFICATION_MODEL = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models_speechbrain")
    return VERIFICATION_MODEL

# ---------------------------
# Evaluation Metrics
# ---------------------------

def compute_cosine_similarity(ref_file, cloned_file):
    # Load pre-trained model
    verification = get_verification_model()
    
    # Load audio files and their sampling rates
    ref_audio, ref_sr = torchaudio.load(ref_file)
    cloned_audio, cloned_sr = torchaudio.load(cloned_file)
    
    # Ensure both audio files have the same sampling rate
    if ref_sr != 16000:
        ref_audio = torchaudio.transforms.Resample(orig_freq=ref_sr, new_freq=16000)(ref_audio)
    if cloned_sr != 16000:
        cloned_audio = torchaudio.transforms.Resample(orig_freq=cloned_sr, new_freq=16000)(cloned_audio)

    # Get speaker embeddings
    ref_embedding = verification.encode_batch(ref_audio)
    cloned_embedding = verification.encode_batch(cloned_audio)
    
    # Compute similarity score
    similarity = verification.similarity(ref_embedding, cloned_embedding)
    return similarity.item()

def analyze_pitch(audio_path, sr=None, f0_min=50.0, f0_max=300.0):
    """
    Compute the mean pitch (F0 in Hz) using librosa's pyin.
    """
    y, sr = librosa.load(audio_path, sr=None)
    f0, voiced_flags, voiced_prob = librosa.pyin(y, sr=sr, fmin=f0_min, fmax=f0_max)
    f0_voiced = f0[~np.isnan(f0)]
    if len(f0_voiced) > 0:
        return float(np.mean(f0_voiced))
    else:
        return None

def compare_mean_pitch(ref_audio_path, cloned_audio_path, sr=None, f0_min=10.0, f0_max=500.0):
    """Compute the difference in mean pitch between two audio files."""
    ref_mean_pitch = analyze_pitch(ref_audio_path, sr=sr, f0_min=f0_min, f0_max=f0_max)
    cloned_mean_pitch = analyze_pitch(cloned_audio_path, sr=sr, f0_min=f0_min, f0_max=f0_max)

    if ref_mean_pitch is not None and cloned_mean_pitch is not None:
        return abs(ref_mean_pitch - cloned_mean_pitch)
    else:
        return None

def pitch_diff_to_similarity(diff):
    """Convert pitch difference to similarity score."""
    if diff is None:
        return None
    diff = max(0.0, diff)
    return 1.0 / (1.0 + diff)

def transcribe_with_whisper(audio_filepath):
    """Transcribe audio using Whisper."""
    model = whisper.load_model("base")
    result = model.transcribe(audio_filepath)
    return result["text"]

def text_similarity(original, transcription):
    """Calculate simple text similarity score."""
    if not transcription:
        return 0.0
    
    original_words = set(original.lower().split())
    transcription_words = set(transcription.lower().split())
    
    if not original_words:
        return 0.0
    
    common_words = original_words.intersection(transcription_words)
    return len(common_words) / len(original_words)

def evaluate_cloned_audio(reference_path, cloned_path, original_text, transcription):
    """Evaluate cloned audio using multiple metrics."""
    # Voice similarity
    cosine_similarity = compute_cosine_similarity(reference_path, cloned_path)
    print(f"Cosine Similarity: {cosine_similarity:.3f}")
    
    # Pitch similarity
    difference = compare_mean_pitch(reference_path, cloned_path)
    pitch_similarity = pitch_diff_to_similarity(difference)    
    print(f"F0 Similarity Score: {pitch_similarity:.3f}")
    
    # Text accuracy
    text_accuracy = text_similarity(original_text, transcription)
    print(f"Text Accuracy: {text_accuracy:.3f}")
    
    # Final combined score (weighted average)
    final_score = (0.4 * cosine_similarity) + (0.3 * pitch_similarity) + (0.3 * text_accuracy)
    return final_score

def get_clone_rewards(self, clip_audio_path: str, clone_text: str, responses: List) -> List[float]:
    """Evaluate miner responses and calculate rewards."""
    print("Evaluating cloned audio...")
    rewards = []
    
    for response in responses:
        if not response.clone_audio:
            print("No cloned audio received.")
            rewards.append(0.0)
            continue
        
        print("Cloned audio received.")
        # Generate a unique timestamp-based filename
        timestamp = int(time.time() * 1000)  # Current time in milliseconds
        clone_audio_path = os.path.join(CLONED_AUDIO_DIR, f"cloned_audio_{timestamp}.wav")
        
        # Save the cloned audio to disk
        with open(clone_audio_path, "wb") as audio_file:
            audio_file.write(base64.b64decode(response.clone_audio))
        
        # Transcribe the cloned audio
        print(f"Transcribing cloned audio: {clone_audio_path}")
        transcription = transcribe_with_whisper(clone_audio_path)
        print(f"Transcription: {transcription}")
        print(f"Original text: {clone_text}")
        
        # Evaluate the cloned audio and calculate final reward
        reward = evaluate_cloned_audio(clip_audio_path, clone_audio_path, clone_text, transcription)
        rewards.append(reward)

    return rewards