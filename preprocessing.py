from pathlib import Path

import librosa
import numpy as np


def load_audio(
    audio_path,
    sample_rate=16000,
    duration=4.0
):
    """
    Load an audio file and convert it into a fixed-length
    mono waveform.

    Parameters
    ----------
    audio_path : str or Path
        Path to the audio file.

    sample_rate : int
        Target sampling rate.

    duration : float
        Target duration in seconds.

    Returns
    -------
    np.ndarray
        Fixed-length audio waveform.
    """

    target_samples = int(sample_rate * duration)

    # Load audio as mono and resample to target sample rate
    audio, _ = librosa.load(
        audio_path,
        sr=sample_rate,
        mono=True
    )

    # Remove DC offset
    audio = audio - np.mean(audio)

    # Normalize safely
    max_value = np.max(np.abs(audio))

    if max_value > 0:
        audio = audio / max_value

    # Fixed-length handling
    if len(audio) < target_samples:

        # Zero padding
        padding = target_samples - len(audio)

        audio = np.pad(
            audio,
            (0, padding),
            mode="constant"
        )

    else:

        # Take the first target_samples
        audio = audio[:target_samples]

    return audio.astype(np.float32)


def extract_mfcc(
    audio,
    sample_rate=16000,
    n_mfcc=40,
    n_fft=1024,
    hop_length=256
):
    """
    Extract MFCC features from an audio waveform.
    """

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )

    return mfcc.astype(np.float32)


def extract_mel_spectrogram(
    audio,
    sample_rate=16000,
    n_mels=64,
    n_fft=1024,
    hop_length=256
):
    """
    Extract log-Mel spectrogram.
    """

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return log_mel.astype(np.float32)


if __name__ == "__main__":

    print("Preprocessing module loaded successfully.")