import os
import numpy as np
from scipy.io import wavfile
from pathlib import Path

# This code reads and merges the drums, bass, and others into an "instrumental" track, 
# and then exports the two layers (vocals and instrumentals) into the folder transport_objects/.

def create_semantic_objects(song_name):
    # Matches the local folder structure: separated/song_name/
    base_path = Path(f"separated/{song_name}")
    out_dir = Path(f"transport_objects/{song_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing vocal and instrumental objects for: {song_name}")
    
    # Print out the path that is being checked (for debugging)
    if not (base_path / "vocals.wav").exists():
        print(f"Error: Could not find files in: {base_path.resolve()}")
        return
    
    # 1. Read stems using SciPy 
    # (opening the audio files, translating the sound waves into matrices)
    sr, vocal = wavfile.read(base_path / "vocals.wav")
    _, drums = wavfile.read(base_path / "drums.wav")
    _, bass = wavfile.read(base_path / "bass.wav")
    _, other = wavfile.read(base_path / "other.wav")
    
    # 2. Combine drums, bass, and other into the Instrumental Object (Si)
    instrumental = drums.astype(float) + bass.astype(float) + other.astype(float)
    
    # Safe signal clipping downcast
    # If np.clip finds any values in the array are too large, it changes the number to 32,767. 
    # If a value is too small, np.clip changes it up to -32,768.
    if vocal.dtype == np.int16:
        instrumental = np.clip(instrumental, -32768, 32767).astype(np.int16)
    elif vocal.dtype == np.int32:
        instrumental = np.clip(instrumental, -2147483648, 2147483647).astype(np.int32)
    else:
        instrumental = instrumental.astype(vocal.dtype)

    # 3. Save files into transport_objects for QUIC streams
    wavfile.write(out_dir / "Sv_vocals.wav", sr, vocal)
    wavfile.write(out_dir / "Si_instrumental.wav", sr, instrumental)
    
    print(f"Successfully created transport layers:")
    print(f"   -> {out_dir}/Sv_vocals.wav")
    print(f"   -> {out_dir}/Si_instrumental.wav")

# Run the process for the tracks
create_semantic_objects("man_i_need")
create_semantic_objects("lamour_de_ma_vie")
