import pyaudio
import numpy as np
import openwakeword
from openwakeword.model import Model

openwakeword.utils.download_models()

# ==========================================
# IMPOSTAZIONI
# ==========================================
MODEL_PATH = "./picsi (1).onnx"
THRESHOLD = 0.1     # Soglia di confidenz a (da 0.0 a 1.0)
CHUNK = 1280         # openWakeWord lavora a blocchi di 80ms (1280 sample)
RATE = 16000         # Frequenza fissa a 16kHz
# ==========================================

def main():
    print("Caricamento del modello in corso...")
    oww_model = Model(wakeword_models=[MODEL_PATH], inference_framework="onnx")
    model_name = list(oww_model.models.keys())[0]
    print(f"Modello '{model_name}' caricato con successo.\n")

    # Inizializzazione dello STREAM audio
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    print("🎤 In ascolto continuo... (Premi Ctrl+C nel terminale per fermare)")

    cooldown = 0
    detections = 0

    try:
        # CICLO INFINITO: Il vero cuore dello streaming continuo
        while True:
            # 1. Legge continuamente i pezzetti di audio (chunk)
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)

            # 2. Li dà in pasto al modello in tempo reale
            prediction = oww_model.predict(audio_data)
            score = prediction[model_name]

            # 3. Se rileva la parola, si "riposa" per circa 1 secondo (cooldown)
            if cooldown > 0:
                cooldown -= 1
            elif score > THRESHOLD:
                detections += 1
                print(f"[{detections}] ✨ Parola rilevata! (Confidenza: {score:.2f})")
                cooldown = 12  # Circa 1 secondo di pausa prima di ascoltare di nuovo la stessa parola

    except KeyboardInterrupt:
        # Questo cattura la combinazione di tasti Ctrl+C
        print("\nArresto manuale richiesto...")

    finally:
        # Chiusura pulita dello stream e rilascio del microfono
        print("\n⏹️ Microfono disattivato.")
        print(f"📊 Totale rilevazioni in questa sessione: {detections}")
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    main()