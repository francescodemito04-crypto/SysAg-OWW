import os
import librosa
import soundfile as sf

# ==========================================
# CONFIGURAZIONE - SCRIPT 1 (CONVERSIONE PCM 16-BIT)
# ==========================================
DIR_INPUT = "1_input_da_convertire"  # Metti qui mp3, flac, m4a, o wav originali
DIR_OUTPUT = "2_output_convertiti"   # Qui finiranno i file .wav PCM 16-bit a 16kHz

TARGET_SR = 16000  # Frequenza standard (16000 Hz)

os.makedirs(DIR_OUTPUT, exist_ok=True)
formati_supportati = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

# ==========================================
# ELABORAZIONE
# ==========================================
print("Inizio la conversione in formato WAV PCM 16-bit (16kHz)...")

for filename in os.listdir(DIR_INPUT):
    if filename.lower().endswith(formati_supportati):
        filepath = os.path.join(DIR_INPUT, filename)
        
        try:
            # Carica l'audio e lo ricampiona a 16000Hz (mono di default)
            samples, sr = librosa.load(filepath, sr=TARGET_SR)
            
            nome_base = os.path.splitext(filename)[0]
            nuovo_nome = f"{nome_base}.wav"
            nuovo_percorso = os.path.join(DIR_OUTPUT, nuovo_nome)
            
            # subtype='PCM_16' forza soundfile a fare la stessa identica conversione 
            # matematica di: (samples * 32767).astype(np.int16)
            sf.write(nuovo_percorso, samples, TARGET_SR, subtype='PCM_16')
            print(f"Convertito in PCM_16: {filename} -> {nuovo_nome}")
            
        except Exception as e:
            print(f"Errore durante la conversione di {filename}: {e}")

print(f"\nFinito! Tutti i file sono pronti in formato PCM 16-bit nella cartella '{DIR_OUTPUT}'.")