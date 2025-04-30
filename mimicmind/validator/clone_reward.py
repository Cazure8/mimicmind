import os
import time
import math
import base64
import torchaudio
import numpy as np
import librosa
import bittensor as bt
from pydub import AudioSegment
from typing import List, Dict, Tuple
from scipy.spatial.distance import cosine
from speechbrain.inference import SpeakerRecognition
import whisper

# Global variables for preloaded models
VERIFICATION_MODEL = None
WHISPER_MODEL = None

# Directory to save cloned audio files
CLONED_AUDIO_DIR = "miner_cloned_voices"
os.makedirs(CLONED_AUDIO_DIR, exist_ok=True)  # Ensure the directory exists

# Configuration constants
QUALITY_WEIGHT = 0.5  # Weight for quality component
LATENCY_WEIGHT = 0.5  # Weight for latency component
QUALITY_THRESHOLD = 0.3  # Minimum quality score to receive full latency reward
TARGET_LATENCY = 10.0  # Target latency in seconds for normalization

# ---------------------------
# Model Loading Functions
# ---------------------------

def get_verification_model():
    """Load the SpeakerRecognition model once and reuse it."""
    global VERIFICATION_MODEL
    if VERIFICATION_MODEL is None:
        VERIFICATION_MODEL = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models_speechbrain")
    return VERIFICATION_MODEL

def get_whisper_model():
    """Load the Whisper model once and reuse it."""
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        WHISPER_MODEL = whisper.load_model("base")
    return WHISPER_MODEL

# ---------------------------
# Evaluation Metrics
# ---------------------------

def compute_cosine_similarity(ref_file, cloned_file):
    """Calculate voice similarity between reference and cloned audio."""
    try:
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
    except Exception as e:
        bt.logging.error(f"Error computing cosine similarity: {e}")
        return 0.0

def analyze_pitch(audio_path, sr=None, f0_min=50.0, f0_max=300.0):
    """Compute the mean pitch (F0 in Hz) using librosa's pyin."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        f0, voiced_flags, voiced_prob = librosa.pyin(y, sr=sr, fmin=f0_min, fmax=f0_max)
        f0_voiced = f0[~np.isnan(f0)]
        if len(f0_voiced) > 0:
            return float(np.mean(f0_voiced))
        else:
            return None
    except Exception as e:
        bt.logging.error(f"Error analyzing pitch: {e}")
        return None

def compare_mean_pitch(ref_audio_path, cloned_audio_path):
    """Compute the difference in mean pitch between two audio files."""
    ref_mean_pitch = analyze_pitch(ref_audio_path)
    cloned_mean_pitch = analyze_pitch(cloned_audio_path)

    if ref_mean_pitch is not None and cloned_mean_pitch is not None:
        return abs(ref_mean_pitch - cloned_mean_pitch)
    else:
        return None

def pitch_diff_to_similarity(diff):
    """Convert pitch difference to similarity score."""
    if diff is None:
        return 0.3  # Default middle value if we can't compute
    diff = max(0.0, diff)
    return 1.0 / (1.0 + diff/10.0)  # Normalize with a characteristic scale of 10Hz

def transcribe_audio(audio_filepath):
    """Transcribe audio using Whisper."""
    try:
        model = get_whisper_model()
        result = model.transcribe(audio_filepath)
        return result["text"]
    except Exception as e:
        bt.logging.error(f"Error transcribing audio: {e}")
        return ""

def text_similarity(original, transcription):
    """Calculate simple text similarity score."""
    if not transcription or not original:
        return 0.0
    
    # Convert to lowercase and split into words
    original_words = set(original.lower().split())
    transcription_words = set(transcription.lower().split())
    
    if not original_words:
        return 0.0
    
    # Calculate Jaccard similarity
    common_words = original_words.intersection(transcription_words)
    union_words = original_words.union(transcription_words)
    
    return len(common_words) / max(1, len(union_words))

def calculate_latency_score(
    uid: int, 
    processing_time: float, 
    response_size: int, 
    min_time: float, 
    max_time: float
) -> float:
    """
    Calculate a latency score that accounts for both absolute and relative performance.
    
    Args:
        uid: Miner's UID
        processing_time: Raw processing time in seconds
        response_size: Size of response in bytes
        min_time: Minimum processing time across miners in this batch
        max_time: Maximum processing time across miners in this batch
        
    Returns:
        Normalized latency score between 0 and 1
    """
    # Guard against division by zero
    if processing_time <= 0:
        return 0.0
        
    # Normalize processing time by response size (larger responses take longer to transmit)
    size_kb = max(1, response_size / 1024)  # Ensure at least 1KB to avoid division by zero
    size_factor = 1.0 + 0.1 * math.log(size_kb)  # Logarithmic scaling for file size
    normalized_time = processing_time / size_factor
    
    # Calculate relative score (how this miner compares to others in the batch)
    relative_score = 0.0
    if max_time > min_time:
        relative_score = 1.0 - ((normalized_time - min_time) / (max_time - min_time))
        relative_score = max(0.0, min(1.0, relative_score))  # Clamp between 0 and 1
    else:
        relative_score = 1.0  # If all times are the same
        
    # Calculate absolute score (reward based on absolute performance)
    absolute_score = math.exp(-normalized_time / TARGET_LATENCY)
    
    # Combined score (70% relative, 30% absolute)
    return 0.7 * relative_score + 0.3 * absolute_score

def evaluate_cloned_audio(
    reference_path: str, 
    cloned_path: str, 
    original_text: str
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate cloned audio using multiple metrics.
    
    Args:
        reference_path: Path to reference audio file
        cloned_path: Path to cloned audio file
        original_text: Original text that should be spoken
        
    Returns:
        Tuple containing quality score and component scores dictionary
    """
    # Get transcription for text accuracy
    transcription = transcribe_audio(cloned_path)
    text_score = text_similarity(original_text, transcription)
    
    # Voice similarity
    voice_score = compute_cosine_similarity(reference_path, cloned_path)
    
    # Pitch similarity
    difference = compare_mean_pitch(reference_path, cloned_path)
    pitch_score = pitch_diff_to_similarity(difference)
    
    # Combined quality score (weighted average)
    quality_score = (0.4 * voice_score) + (0.3 * pitch_score) + (0.3 * text_score)
    
    # Create dictionary of component scores for logging
    component_scores = {
        "voice_similarity": voice_score,
        "pitch_similarity": pitch_score,
        "text_accuracy": text_score,
        "transcription": transcription
    }
    
    return quality_score, component_scores

def get_clone_rewards(
    self, 
    clip_audio_path: str, 
    clone_text: str, 
    responses: List,
    miner_uids: List[int],
    processing_times: Dict[int, float],
    response_sizes: Dict[int, int]
) -> np.ndarray:
    """
    Evaluate miner responses with both quality and latency metrics.
    The final reward will be redistributed by rank in the validator.
    
    Args:
        clip_audio_path: Path to reference audio
        clone_text: Text to clone
        responses: List of miner responses
        miner_uids: List of miner UIDs
        processing_times: Dictionary mapping UIDs to processing times
        response_sizes: Dictionary mapping UIDs to response sizes in bytes
        
    Returns:
        Array of rewards for each miner
    """
    bt.logging.info("Evaluating cloned audio with quality and latency metrics...")
    
    # Initialize arrays for tracking metrics
    quality_scores = np.zeros(len(responses))
    latency_scores = np.zeros(len(responses))
    final_rewards = np.zeros(len(responses))
    
    # Collect metrics for each response
    valid_times = []
    valid_sizes = []
    valid_indices = []
    
    # First pass: calculate quality scores and collect valid timing data
    for idx, response in enumerate(responses):
        uid = miner_uids[idx]
        
        # Handle missing responses
        if not response.clone_audio:
            bt.logging.debug(f"Miner {uid}: No cloned audio received.")
            quality_scores[idx] = 0.0
            latency_scores[idx] = 0.0
            continue
        
        # Save the cloned audio for evaluation
        timestamp = int(time.time() * 1000)
        clone_audio_path = os.path.join(CLONED_AUDIO_DIR, f"cloned_audio_{timestamp}.wav")
        
        try:
            with open(clone_audio_path, "wb") as audio_file:
                audio_file.write(base64.b64decode(response.clone_audio))
            
            # Evaluate audio quality
            quality_score, component_scores = evaluate_cloned_audio(
                clip_audio_path, 
                clone_audio_path, 
                clone_text
            )
            
            # Store quality score
            quality_scores[idx] = quality_score
            
            # Log detailed component scores
            bt.logging.debug(
                f"Miner {uid} Quality: {quality_score:.3f} (Voice: {component_scores['voice_similarity']:.3f}, "
                f"Pitch: {component_scores['pitch_similarity']:.3f}, Text: {component_scores['text_accuracy']:.3f})"
            )
            
            # Collect valid processing times and sizes
            if uid in processing_times and processing_times[uid] > 0:
                valid_times.append(processing_times[uid])
                valid_sizes.append(response_sizes.get(uid, 0))
                valid_indices.append(idx)
            
        except Exception as e:
            bt.logging.error(f"Error processing response from miner {uid}: {e}")
            quality_scores[idx] = 0.0
            latency_scores[idx] = 0.0
    
    # Second pass: normalize latency scores across all miners in this batch
    if valid_times:
        min_time = min(valid_times)
        max_time = max(valid_times)
        
        for idx in valid_indices:
            uid = miner_uids[idx]
            latency_scores[idx] = calculate_latency_score(
                uid,
                processing_times[uid],
                response_sizes.get(uid, 0),
                min_time,
                max_time
            )
            
            # Apply quality gate - reduce latency component if quality is poor
            if quality_scores[idx] < QUALITY_THRESHOLD:
                # Scale latency score based on quality ratio to threshold
                quality_ratio = quality_scores[idx] / QUALITY_THRESHOLD
                latency_scores[idx] *= quality_ratio
            
            bt.logging.debug(
                f"Miner {uid} Latency: {latency_scores[idx]:.3f} (Time: {processing_times[uid]:.3f}s, "
                f"Size: {response_sizes.get(uid, 0)} bytes)"
            )
    
    # Calculate final rewards - combined quality and latency scores
    for idx in range(len(responses)):
        # Combined score with equal weighting for quality and latency
        final_rewards[idx] = (QUALITY_WEIGHT * quality_scores[idx]) + (LATENCY_WEIGHT * latency_scores[idx])
        uid = miner_uids[idx]
        
        bt.logging.info(
            f"Miner {uid} Final Score: {final_rewards[idx]:.4f} "
            f"(Quality: {quality_scores[idx]:.3f}, Latency: {latency_scores[idx]:.3f})"
        )
    
    return final_rewards