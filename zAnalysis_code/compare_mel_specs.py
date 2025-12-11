"""Compare Lingvo vs Librosa mel spectrogram parameters"""
import numpy as np
import librosa

# Simulate comparison
sample_rate = 16000
duration = 1.0  # 1 second
audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)

# LIBROSA configuration (current code)
librosa_params = {
    'n_fft': 512,
    'hop_length': 160,  # 10ms
    'win_length': 400,  # 25ms  
    'window': 'hamming',
    'n_mels': 128,
    'fmin': 0,
    'fmax': 8000
}

# LINGVO configuration (from frontend.py)
lingvo_params = {
    'n_fft': 512,  # _NextPowerOfTwo(400) = 512, but can be doubled with fft_overdrive
    'hop_length': 160,  # 10ms = 16000 * 0.01 = 160
    'win_length': 401,  # 25ms + 1 for preemph = int(16000 * 0.025) + 1 = 401
    'window': 'hann',  # Lingvo uses HANNING (not hamming!)
    'n_mels': 128,
    'fmin': 125.0,  # Lingvo default (not 0!)
    'fmax': 7600.0  # Lingvo default (not 8000!)
}

print("=" * 80)
print("CRITICAL DIFFERENCES FOUND:")
print("=" * 80)
print()
print("1. WINDOW FUNCTION:")
print(f"   Librosa: {librosa_params['window']} (Hamming window)")
print(f"   Lingvo:  {lingvo_params['window']} (Hann/Hanning window)")
print("   → Different filter shapes!")
print()
print("2. FREQUENCY RANGE:")
print(f"   Librosa: fmin={librosa_params['fmin']}, fmax={librosa_params['fmax']}")
print(f"   Lingvo:  fmin={lingvo_params['fmin']}, fmax={lingvo_params['fmax']}")
print("   → Different mel filterbank coverage!")
print()
print("3. WINDOW LENGTH:")
print(f"   Librosa: {librosa_params['win_length']} samples")
print(f"   Lingvo:  {lingvo_params['win_length']} samples (+1 for preemphasis)")
print()
print("4. FFT SIZE (if fft_overdrive=True in Lingvo):")
print(f"   Librosa: n_fft={librosa_params['n_fft']}")
print(f"   Lingvo:  n_fft=512 or 1024 (doubled if fft_overdrive=True)")
print()
print("5. PREEMPHASIS:")
print("   Librosa: No built-in preemphasis")
print("   Lingvo:  Applies preemph=0.97 coefficient by default")
print("   → Lingvo applies: signal[n] - 0.97*signal[n-1]")
print()

# Test actual mel spectrogram computation
print("=" * 80)
print("TESTING ACTUAL OUTPUT DIFFERENCES:")
print("=" * 80)

# Librosa mel spec (current)
mel_librosa = librosa.feature.melspectrogram(
    y=audio, sr=sample_rate, **librosa_params, power=2.0)

# Lingvo-like mel spec
# Apply preemphasis manually
audio_preemph = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

mel_lingvo_like = librosa.feature.melspectrogram(
    y=audio_preemph, sr=sample_rate,
    n_fft=512, hop_length=160, win_length=401,
    window='hann', n_mels=128, fmin=125.0, fmax=7600.0, power=2.0)

print(f"\nLibrosa mel spec shape: {mel_librosa.shape}")
print(f"Librosa mean: {mel_librosa.mean():.6f}, std: {mel_librosa.std():.6f}")
print(f"Librosa range: [{mel_librosa.min():.6f}, {mel_librosa.max():.6f}]")
print()
print(f"Lingvo-like mel spec shape: {mel_lingvo_like.shape}")  
print(f"Lingvo-like mean: {mel_lingvo_like.mean():.6f}, std: {mel_lingvo_like.std():.6f}")
print(f"Lingvo-like range: [{mel_lingvo_like.min():.6f}, {mel_lingvo_like.max():.6f}]")
print()

# Compute difference
diff = np.abs(mel_librosa - mel_lingvo_like).mean()
rel_diff = diff / (mel_librosa.mean() + 1e-8) * 100
print(f"Mean absolute difference: {diff:.6f}")
print(f"Relative difference: {rel_diff:.2f}%")
print()

print("=" * 80)
print("RECOMMENDATION:")
print("=" * 80)
print()
print("The MBT model was likely trained with Lingvo's audio frontend, which uses:")
print("  1. Hann window (not Hamming)")
print("  2. Frequency range: 125-7600 Hz (not 0-8000 Hz)")
print("  3. Preemphasis coefficient: 0.97")
print("  4. Window length: 401 samples (25ms + 1 for preemph)")
print()
print("Try updating your preprocessing to match these Lingvo defaults!")
