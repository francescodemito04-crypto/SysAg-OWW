#!/usr/bin/env python3
"""
Verifica Dataset PIXIE con WhisperX — batching reale su GPU/CPU.
Salva il risultato del caricamento audio in un file cache (.pkl)
per saltare la fase di caricamento nelle esecuzioni successive.

Modalità disponibili:
  1 — Positivi   (keyword: pixie / pixi / piksi / picsi)
  2 — Negativi   (keyword: Pizza, Pixel, Pyrex, Taxi, Dixit, Dixie, Pasticcio, Bixby, Pipsi, Mix)
"""

import os
import sys
import glob
import warnings
import re
import shutil
import gc
import pickle

import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ==============================================================================
#  FIX DLL WINDOWS — carica le DLL CUDA dalla venv prima di importare whisperx
# ==============================================================================
if sys.platform == "win32":
    import ctypes
    import site

    _dll_names = [
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cudnn_ops_infer64_8.dll",
        "cudnn64_8.dll",
    ]

    _search_dirs = []
    for _sp in site.getsitepackages():
        for _root, _dirs, _files in os.walk(_sp):
            if any(f.endswith(".dll") for f in _files):
                _search_dirs.append(_root)

    for _dll in _dll_names:
        for _d in _search_dirs:
            _path = os.path.join(_d, _dll)
            if os.path.exists(_path):
                try:
                    ctypes.CDLL(_path)
                except Exception:
                    pass
                break
# ==============================================================================

# ==============================================================================
#  PROFILI DI VERIFICA
# ==============================================================================

PROFILI = {
    "1": {
        "nome":   "Positivi",
        "prompt": "Picsi",
        "regex":  re.compile(r"(pixie|pixi|piksi|picsi)", re.IGNORECASE),
    },
    "2": {
        "nome":   "Negativi / Avversi",
        "prompt": "Pizza, Pixel, Pyrex, Taxi, Dixit, Dixie, Pasticcio, Bixby, Pepsi, Pipsi, Mix",
        "regex":  re.compile(
            r"(Pizza|Pixel|Pyrex|Taxi|Dixit|Dixie|Pasticcio|Bixby|Pepsi|Pipsi|Mix)",
            re.IGNORECASE
        ),
    },
}

# ==============================================================================
#  CONFIGURAZIONE COMUNE — modifica solo questa sezione
# ==============================================================================

INPUT_DIR    = "audio_input"
DISCARD_DIR  = "audio_scartati"
REPORT_FILE  = "report_falliti.txt"
CACHE_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_cache.pkl")

MODELLO      = "base"    # tiny | base | small | medium | large-v2 | large-v3
BATCH_SIZE   = 32        # Riduci a 4-8 se poca VRAM
COMPUTE_TYPE = "auto"    # "auto" | "float16" | "int8" | "float32"
LINGUA       = "it"

# ==============================================================================


def scegli_profilo():
    print("=" * 60)
    print(" VERIFICA DATASET PIXIE — Seleziona modalità")
    print("=" * 60)
    for chiave, p in PROFILI.items():
        print(f"  {chiave}) {p['nome']}")
    print()

    while True:
        scelta = input("Inserisci il numero della modalità: ").strip()
        if scelta in PROFILI:
            profilo = PROFILI[scelta]
            print(f"\n[✓] Modalità selezionata: {profilo['nome']}\n")
            return profilo
        print(f"  ⚠️  Scelta non valida. Inserisci uno tra: {', '.join(PROFILI)}")


def carica_audio_batch(file_paths):
    import whisperx
    risultati = []
    print(f"[*] Caricamento di {len(file_paths)} file audio...")
    for fp in tqdm(file_paths, desc="Caricamento audio", unit="file"):
        try:
            audio = whisperx.load_audio(fp)
            risultati.append((os.path.basename(fp), fp, audio))
        except Exception as e:
            print(f"  ⚠️  Impossibile leggere {os.path.basename(fp)}: {e}")
            risultati.append((os.path.basename(fp), fp, None))

    print(f"[*] Salvataggio cache in '{CACHE_FILE}'...")
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(risultati, f)
    print(f"[✓] Cache salvata ({os.path.getsize(CACHE_FILE) / 1024 / 1024:.1f} MB).")

    return risultati


def carica_da_cache():
    print(f"[*] Cache trovata '{CACHE_FILE}' — salto il caricamento audio.")
    with open(CACHE_FILE, "rb") as f:
        risultati = pickle.load(f)
    print(f"[✓] Caricati {len(risultati)} file dalla cache.")
    return risultati


def trascrivi_e_verifica(model, audio_entries, target_regex):
    file_passati = []
    file_falliti = []

    validi   = [(n, fp, a) for n, fp, a in audio_entries if a is not None]
    invalidi = [(n, fp)    for n, fp, a in audio_entries if a is None]

    for nome, fp in invalidi:
        file_falliti.append((nome, "Errore lettura file audio"))
        try:
            shutil.move(fp, os.path.join(DISCARD_DIR, nome))
        except Exception:
            pass

    totale     = len(audio_entries)
    processati = len(invalidi)

    for nome, fp, audio in validi:
        processati += 1
        try:
            risultato = model.transcribe(
                audio,
                batch_size=BATCH_SIZE,
                language=LINGUA,
                print_progress=False
            )

            testo_completo = " ".join(
                seg.get("text", "").strip()
                for seg in risultato.get("segments", [])
            ).strip()

            testo_pulito = re.sub(r"[^\w\s]", "", testo_completo).strip().lower()

            if target_regex.search(testo_pulito):
                tqdm.write(f"[{processati}/{totale}] {nome} ... ✅ PASSED")
                file_passati.append(nome)
            else:
                tqdm.write(f"[{processati}/{totale}] {nome} ... ❌ FAILED (Ha capito: '{testo_completo}')")
                file_falliti.append((nome, testo_completo))
                shutil.move(fp, os.path.join(DISCARD_DIR, nome))

        except Exception as e:
            tqdm.write(f"[{processati}/{totale}] {nome} ... ⚠️  ERRORE ({e})")
            file_falliti.append((nome, f"Errore tecnico: {e}"))
            try:
                shutil.move(fp, os.path.join(DISCARD_DIR, nome))
            except Exception:
                pass

    return file_passati, file_falliti


def salva_report(file_falliti):
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("=== LOG FILE NON RICONOSCIUTI DA WHISPERX ===\n")
        f.write(f"Spostati nella cartella '{DISCARD_DIR}'.\n")
        f.write("Troppo distorti dall'augmentation o con rumore che copre la voce.\n\n")
        for nome, capito in file_falliti:
            f.write(f"FILE: {nome}\n")
            f.write(f"WHISPERX HA CAPITO: {capito}\n")
            f.write("-" * 40 + "\n")


def main():
    try:
        import whisperx
    except ImportError:
        print("[ERR] whisperx non installato. Esegui: pip install whisperx")
        sys.exit(1)

    profilo = scegli_profilo()

    if not os.path.isdir(INPUT_DIR):
        print(f"[ERR] La cartella '{INPUT_DIR}' non esiste.")
        sys.exit(1)

    os.makedirs(DISCARD_DIR, exist_ok=True)

    file_wav = glob.glob(os.path.join(INPUT_DIR, "*.wav"))
    if not file_wav:
        print(f"[!] Nessun file .wav trovato in '{INPUT_DIR}'.")
        sys.exit(0)

    print(f"[*] Trovati {len(file_wav)} file da verificare.")

    if torch.cuda.is_available():
        device       = "cuda"
        compute_type = "float16" if COMPUTE_TYPE == "auto" else COMPUTE_TYPE
        print(f"✅ GPU trovata: {torch.cuda.get_device_name(0)}")
        print(f"   Modello: {MODELLO}  |  Compute type: {compute_type}  |  Batch size: {BATCH_SIZE}")
    else:
        device       = "cpu"
        compute_type = "int8" if COMPUTE_TYPE == "auto" else COMPUTE_TYPE
        print(f"⚠️  Nessuna GPU. Utilizzo CPU con compute_type={compute_type}.")

    if os.path.exists(CACHE_FILE):
        audio_entries = carica_da_cache()
    else:
        audio_entries = carica_audio_batch(file_wav)

    print(f"\n[*] Caricamento modello WhisperX '{MODELLO}'...")
    model = whisperx.load_model(
        MODELLO,
        device=device,
        compute_type=compute_type,
        language=LINGUA,
        asr_options={
            "initial_prompt": profilo["prompt"],
            "temperatures": [0.0],
        }
    )
    print("[*] Modello caricato.\n")

    print(f"[*] Inizio verifica batch (batch_size={BATCH_SIZE})...\n")
    file_passati, file_falliti = trascrivi_e_verifica(model, audio_entries, profilo["regex"])

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print(f" VERIFICA COMPLETATA [{profilo['nome']}] — REPORT FINALE")
    print("=" * 60)
    print(f"Totale file analizzati : {len(file_wav)}")
    print(f"✅ File SUPERATI       : {len(file_passati)}")
    print(f"❌ File FALLITI        : {len(file_falliti)}  (spostati in '{DISCARD_DIR}')")

    if file_falliti:
        salva_report(file_falliti)
        print(f"\n[!] Log salvato in '{REPORT_FILE}'.")
        print(f"[✓] '{INPUT_DIR}' contiene ora solo il dataset pulito!")
    else:
        print("\n🎉 Tutti i file hanno superato la verifica. Il dataset è perfetto!")

    if os.path.exists(CACHE_FILE):
        print(f"\n💡 Cache audio disponibile in '{CACHE_FILE}'.")
        print(f"   Eliminala se aggiungi nuovi file in '{INPUT_DIR}'.")


if __name__ == "__main__":
    main()