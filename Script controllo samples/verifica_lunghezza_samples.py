import os
import shutil
import soundfile as sf

# Inserisci qui il nome della cartella dove hai le vocali
DIR_VOCALI = "input_vocali" 
DIR_SCARTATI = "input_scartati" 

# Controlla se la cartella degli scartati esiste, altrimenti la crea
if not os.path.exists(DIR_SCARTATI):
    os.makedirs(DIR_SCARTATI)
    print(f" Cartella '{DIR_SCARTATI}' creata.")

esatti = 0
anomali = 0

print(f"Analisi dei file in '{DIR_VOCALI}' in corso...\n")

for nome_file in os.listdir(DIR_VOCALI):
    if nome_file.lower().endswith(".wav"):
        percorso = os.path.join(DIR_VOCALI, nome_file)
        
        # Legge i metadati istantaneamente
        info = sf.info(percorso)
        durata = info.duration
        
        # Usiamo un margine di tolleranza. A causa del campionamento matematico, 
        # un file da 1 secondo perfetto potrebbe risultare lungo 0.999 o 1.001
        if 0.99 <= durata <= 1.01:
            esatti += 1
        else:
            print(f"⚠️ ANOMALIA: '{nome_file}' dura {durata:.3f} secondi. Spostamento in corso...")
            
            # Calcola il nuovo percorso dove spostare il file
            percorso_destinazione = os.path.join(DIR_SCARTATI, nome_file)
            
            # Sposta fisicamente il file dalla cartella originale a quella degli scartati
            shutil.move(percorso, percorso_destinazione)
            
            anomali += 1

print("\n--- REPORT FINALE ---")
print(f"✅ File perfetti (~1 secondo) rimasti in '{DIR_VOCALI}': {esatti}")
print(f"❌ File con durata errata spostati in '{DIR_SCARTATI}': {anomali}")