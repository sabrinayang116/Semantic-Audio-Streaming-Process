import os
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt
from pydub import AudioSegment

# This function builds a filter that only lets a specific middle band of frequencies pass through
def butter_bandpass(lowcut, highcut, fs, order=5):
    # Calculate the Nyquist frequency, which is exactly half of our audio sampling rate
    nyq = 0.5 * fs
    # Express the lower frequency cutoff as a ratio relative to the Nyquist limit
    low = lowcut / nyq
    # Express the higher frequency cutoff as a ratio relative to the Nyquist limit
    high = highcut / nyq
    # Create and return the stable digital filter coefficients using a bandpass design
    return butter(order, [low, high], btype='band', output='sos')

def process_hybrid_split(input_dir="input", output_dir="transport_objects"):
    """
    Decodes a stereo MP3, isolates the center channel via mid/side extraction 
    and bandpass filtering for vocals, and uses L-R phase cancellation for instrumentals.
    """
    # CHANGE THIS LINE: Make sure it has the underscore to match your file exactly!
    song_name = "the_cure"
    
    # This correctly assembles the path: input/the_cure.mp3
    input_path = Path(input_dir) / f"{song_name}.mp3"
    
    # This sets the output folder target to: transport_objects/the_cure
    song_output_folder = Path(output_dir) / song_name
    
    # Define custom names to prevent overriding any previous files
    vocal_output_path = song_output_folder / "sv_vocal_hybrid.wav"
    inst_output_path = song_output_folder / "si_instrumental_hybrid.wav"
    wav_reference_path = song_output_folder / "original_reference_stereo.wav"
    
    # Check if the raw MP3 file exists on disk
    if not input_path.exists():
        print(f"[-] Missing input file at {input_path}. Ensure the file is named 'the_cure.mp3'.")
        return

    print(f"\nProcessing: {song_name}.mp3")
    print(f"├── Decoding stereo MP3 bitstream...")
    
    # Load the MP3 file and keep it in Stereo (2 channels) to perform phase cancellation
    audio = AudioSegment.from_mp3(str(input_path))
    if audio.channels < 2:
        print("[-] Error: Phase cancellation requires a stereo file. This file is mono.")
        return
        
    fs = audio.frame_rate
    
    # Convert raw binary bytes into a NumPy array, then reshape it into a 2D matrix [Samples, Channels]
    raw_samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    stereo_data = raw_samples.reshape((-1, audio.channels))
    
    # Separate the left and right channels into independent 1D arrays
    left_channel = stereo_data[:, 0]
    right_channel = stereo_data[:, 1]
    
    # Scale both channels down so values sit safely between -1.0 and 1.0
    max_val = max(np.max(np.abs(left_channel)), np.max(np.abs(right_channel)))
    if max_val > 0:
        left_channel /= max_val
        right_channel /= max_val

    # --- INSTRUMENTAL EXTRACTION (Stereo Phase Cancellation) ---
    print(f"├── Canceling center channel (L - R) to extract instrumentals...")
    # Subtracting right from left causes perfectly centered signals (vocals) to cancel out entirely
    inst_data = left_channel - right_channel

    # --- VOCAL EXTRACTION (Center Channel Isolation + Bandpass Filter) ---
    print(f"├── Isolating mid-channel and applying vocal bandpass filter...")
    # Summing left and right accentuates everything mixed dead-center (Mid signal)
    mid_channel = (left_channel + right_channel) / 2.0
    
    # Set vocal frequency limits
    vocal_low = 150.0
    vocal_high = 4000.0
    
    # Generate the filter instructions and run them over the center-channel signal
    vocal_sos = butter_bandpass(vocal_low, vocal_high, fs, order=6)
    vocal_data = sosfiltfilt(vocal_sos, mid_channel)

    # --- GAIN NORMALIZATION ---
    # Boost or tame both streams independently so their peaks reach exactly 1.0
    if np.max(np.abs(vocal_data)) > 0:
        vocal_data = vocal_data / np.max(np.abs(vocal_data))
    if np.max(np.abs(inst_data)) > 0:
        inst_data = inst_data / np.max(np.abs(inst_data))

    # Safely build the output directory structure if it doesn't already exist
    os.makedirs(song_output_folder, exist_ok=True)
    
    # Save the original stereo mix as a reference file
    normalized_stereo = (stereo_data / max_val * 32767).astype(np.int16)
    wavfile.write(wav_reference_path, fs, normalized_stereo)
    
    # Save the hybrid files out to disk as mono 16-bit PCM WAV tracks
    wavfile.write(vocal_output_path, fs, (vocal_data * 32767).astype(np.int16))
    wavfile.write(inst_output_path, fs, (inst_data * 32767).astype(np.int16))
    print(f"└── Successfully wrote files to {song_output_folder}/")

if __name__ == "__main__":
    print("=========================================================================")
    print("         HYBRID DSP: STEREO CANCELLATION + LINEAR FILTERING              ")
    print("=========================================================================")
    
    process_hybrid_split()
        
    print("\n=========================================================================")