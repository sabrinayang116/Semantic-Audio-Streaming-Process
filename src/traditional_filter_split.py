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

# This function builds a filter that punches a hole in the middle, blocking a specific band of frequencies
def butter_bandreject(lowcut, highcut, fs, order=5):
    # Calculate the Nyquist frequency, which is exactly half of our audio sampling rate
    nyq = 0.5 * fs
    # Express the lower frequency cutoff as a ratio relative to the Nyquist limit
    low = lowcut / nyq
    # Express the higher frequency cutoff as a ratio relative to the Nyquist limit
    high = highcut / nyq
    # Create and return the stable digital filter coefficients using a band-reject design
    return butter(order, [low, high], btype='bandstop', output='sos')

def process_mp3_input(song_name, input_dir="input", output_dir="transport_objects"):
    """
    Decodes an MP3 input file, converts it to float32 mono array,
    applies traditional filters, and outputs distinct baseline WAV files.
    """
    # Assemble the physical file path pointing to our incoming MP3 track
    input_path = Path(input_dir) / f"{song_name}.mp3"
    # Construct the destination folder path where our split audio objects will live
    song_output_folder = Path(output_dir) / song_name
    
    # Establish the unique path for the traditional filter vocal baseline to protect the AI version
    vocal_output_path = song_output_folder / "Sv_vocals_traditional.wav"
    # Establish the unique path for the traditional filter instrumental baseline to protect the AI version
    inst_output_path = song_output_folder / "Si_instrumental_traditional.wav"
    # Establish the precise file path where the uncompressed original reference WAV will be saved
    wav_reference_path = song_output_folder / "original_reference.wav"
    
    # Check if the raw MP3 file actually exists on the disk before moving forward
    if not input_path.exists():
        print(f"[-] Missing input file at {input_path}. Skipping {song_name}.")
        return

    print(f"\nProcessing: {song_name}.mp3")
    print(f"Decoding MP3 bitstream to PCM...")
    
    # Read the compressed MP3 file from disk and instantly force it into a single audio channel (Mono)
    audio = AudioSegment.from_mp3(str(input_path)).set_channels(1)
    # Extract the physical sampling rate of the audio, which tells us samples per second (e.g., 44100Hz)
    fs = audio.frame_rate
    
    # Convert the raw, uncompressed binary audio bytes into a standard numerical NumPy array
    data = np.array(audio.get_array_of_samples(), dtype=np.float32)
    
    # Divide the entire array by its maximum absolute value so all numbers scale perfectly between -1.0 and 1.0
    data = data / np.max(np.abs(data))

    # Set the lower frequency boundary for human speech (cuts out deep bass rumble below 150 vibrations per second)
    vocal_low = 150.0
    # Set the upper frequency boundary for human speech (cuts out sharp cymbal sizzle above 4000 vibrations per second)
    vocal_high = 4000.0

    print(f"├── Applying Baseline Bandpass Filter ({vocal_low}Hz - {vocal_high}Hz)...")
    # Generate the custom mathematical instructions (coefficients) for our vocal isolation filter
    vocal_sos = butter_bandpass(vocal_low, vocal_high, fs, order=6)
    # Run the audio data through the filter forward and backward to prevent any timing delays or phase shifts
    vocal_data = sosfiltfilt(vocal_sos, data)
    
    print(f"├── Applying Baseline Band-Reject Filter...")
    # Generate the custom mathematical instructions to block out the core vocal frequency range
    inst_sos = butter_bandreject(vocal_low, vocal_high, fs, order=6)
    # Filter the audio data to keep only the extreme lows and highs, leaving an empty pocket in the middle
    inst_data = sosfiltfilt(inst_sos, data)

    # Check if our isolated vocal track contains any sound data at all
    if np.max(np.abs(vocal_data)) > 0:
        # Boost or tame the vocal waveform so its absolute peak amplitude reaches exactly 1.0
        vocal_data = vocal_data / np.max(np.abs(vocal_data))
    # Check if our isolated instrumental track contains any sound data at all
    if np.max(np.abs(inst_data)) > 0:
        # Boost or tame the instrumental waveform so its absolute peak amplitude reaches exactly 1.0
        inst_data = inst_data / np.max(np.abs(inst_data))

    # Safely build the output directory structure if it doesn't already exist on your computer
    os.makedirs(song_output_folder, exist_ok=True)
    
    # Scale our -1.0 to 1.0 float numbers back up to 16-bit integer bounds and save the original reference mix
    wavfile.write(wav_reference_path, fs, (data * 32767).astype(np.int16))
    
    # Convert and write out the traditional vocal benchmark into its new dedicated file name
    wavfile.write(vocal_output_path, fs, (vocal_data * 32767).astype(np.int16))
    # Convert and write out the traditional instrumental benchmark into its new dedicated file name
    wavfile.write(inst_output_path, fs, (inst_data * 32767).astype(np.int16))
    print(f"└── Successfully added traditional baselines to {song_output_folder}/")

if __name__ == "__main__":
    # Define the precise list of folders matching your specific music track names
    target_songs = ["man_i_need", "lamour_de_ma_vie"]
    
    print("=========================================================================")
    print("     DSP BASELINE: MP3 TO SEPARATE TRADITIONAL WAV FILES CONVERTER       ")
    print("=========================================================================")
    
    # Loop through our track list one by one and run them through our filtering process
    for song in target_songs:
        process_mp3_input(song)
        
    print("\n=========================================================================")