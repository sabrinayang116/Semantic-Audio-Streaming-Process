import os
import numpy as np
from scipy.io import wavfile
from pathlib import Path

def simulate_transport(song_name, loss_rate=0.07):
    print(f"\nStarting QUIC Transport Simulation for: {song_name} ({int(loss_rate*100)}% Packet Loss)")
    
    in_dir = Path(f"transport_objects/{song_name}")
    out_dir = Path(f"simulated_output/{song_name}_loss_{int(loss_rate*100)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load the pristine Semantic Objects
    sr, vocal = wavfile.read(in_dir / "Sv_vocals.wav")
    _, instrumental = wavfile.read(in_dir / "Si_instrumental.wav")
    
    # Convert raw arrays to raw bytes for packetization
    v_bytes = vocal.tobytes()
    i_bytes = instrumental.tobytes()
    
    # Chunk sizing matching QUIC standard payload maximums
    PACKET_SIZE = 1200 
    v_chunks = [v_bytes[x:x+PACKET_SIZE] for x in range(0, len(v_bytes), PACKET_SIZE)]
    i_chunks = [i_bytes[x:x+PACKET_SIZE] for x in range(0, len(i_bytes), PACKET_SIZE)]
    
    print(f"   ↳ Segmented into Packets -> Stream 0 (Vocals): {len(v_chunks)} | Stream 1 (Instruments): {len(i_chunks)}")
    
    # 2. Simulate Network Drop Vectors (Uniform Random Erasure Mask)
    np.random.seed(42) # Seeded for scientific reproducibility across EEP vs UEP sweeps
    v_loss_mask = np.random.rand(len(v_chunks)) < loss_rate
    i_loss_mask = np.random.rand(len(i_chunks)) < loss_rate
    
    # 3. Apply Your Proposed UEP Decoding Math
    # Stream 0 (Vocals) features 30% heavy FEC recovery capability
    # Stream 1 (Instruments) features minimal 4% recovery protection
    v_fec_recovery_threshold = 0.30
    i_fec_recovery_threshold = 0.04
    
    v_reconstructed = []
    for idx, chunk in enumerate(v_chunks):
        if v_loss_mask[idx]:
            # If the loss bursts are within our 30% FEC threshold boundary, recover it!
            if loss_rate <= v_fec_recovery_threshold:
                v_reconstructed.append(chunk) # Perfect Recovery
            else:
                v_reconstructed.append(b'\x00' * len(chunk)) # Unrecoverable Dropout (Concealment gap)
        else:
            v_reconstructed.append(chunk)
            
    i_reconstructed = []
    for idx, chunk in enumerate(i_chunks):
        if i_loss_mask[idx]:
            if loss_rate <= i_fec_recovery_threshold:
                i_reconstructed.append(chunk)
            else:
                i_reconstructed.append(b'\x00' * len(chunk)) # Stutter/Drop
        else:
            i_reconstructed.append(chunk)
            
    # 4. Reconstruct byte frames back to audio arrays
    v_final = np.frombuffer(b''.join(v_reconstructed), dtype=vocal.dtype)
    i_final = np.frombuffer(b''.join(i_reconstructed), dtype=instrumental.dtype)
    
    # Ensure time alignment array lengths match perfectly
    min_len = min(len(v_final), len(i_final))
    v_final, i_final = v_final[:min_len], i_final[:min_len]
    
    # 5. Presentation Time Stamp (PTS) Coherence Condition Mix: AudioOut = Sv + Si
    audio_out = v_final.astype(float) + i_final.astype(float)
    
    # Downcast and clip safely back to native depth
    if vocal.dtype == np.int16:
        audio_out = np.clip(audio_out, -32768, 32767).astype(np.int16)
    else:
        audio_out = audio_out.astype(vocal.dtype)
        
    # Export degraded tracks for ViSQOL automated listen tests
    wavfile.write(out_dir / "reconstructed_mix.wav", sr, audio_out)
    print(f"Simulation complete. Output stored in: {out_dir}/reconstructed_mix.wav")

    wavfile.write(out_dir / "rec_vocals.wav", sr, v_final.astype(np.int16))
    wavfile.write(out_dir / "rec_instrumental.wav", sr, i_final.astype(np.int16))

# Run the experimental sweep requirements matrix across both processed tracks
for loss in [0.00, 0.07, 0.12]:
    simulate_transport("man_i_need", loss_rate=loss)
    simulate_transport("lamour_de_ma_vie", loss_rate=loss)
