import streamlit as st
import numpy as np
from pydub import AudioSegment
from scipy.signal import butter, sosfilt
import wave
import os
import tempfile
from scipy.io.wavfile import write as wav_write

# Configuracion inicial
st.set_page_config(page_title="MTRS Generator", layout="centered")
st.title("🎧 MTRS Sound Therapy Generator")
st.write("Selecciona las características del tinnitus para generar un audio terapéutico personalizado.")

# Entrada del usuario
freq = st.selectbox("Frecuencia del tinnitus (Hz)", [250, 500, 1000, 2000, 4000, 6000, 8000])
db = st.slider("Volumen percibido (dB HL)", 0, 80, 40)

# Generar sonido puro
def dbhl_to_amplitude(db_hl):
    return 10 ** ((db_hl - 80) / 20)

def generate_tone(frequency, duration=1.0, fs=44100, db_hl=40):
    amplitude = dbhl_to_amplitude(db_hl)
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    tone = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    return tone, fs

# Boton para reproducir tono puro
if st.button("🔊 Probar tono puro"):
    tone, fs = generate_tone(freq, db_hl=db)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav_write(tmp.name, fs, (tone * 32767).astype(np.int16))
    st.audio(tmp.name, format="audio/wav")

# Cargar audio base
import os
audio_path = os.path.join(os.path.dirname(__file__), "static", "sonido_lluvia_short.wav")
audio = AudioSegment.from_file(audio_path, format="wav")
samples = np.array(audio.get_array_of_samples()).astype(np.float32)
fs = audio.frame_rate

# Procesamiento MTRS
# Se calcula un ancho de banda del 26% ~1/3 de octava (estandar para terapias auditivas). Se define una banda por encima y por debajo
def get_mtrs_bands(center_freq):
    band_width = center_freq * 0.26
    lower_band = (center_freq - band_width, center_freq - band_width / 2)
    upper_band = (center_freq + band_width / 2, center_freq + band_width)
    return lower_band, upper_band

# Aumenta volumen de banda de frecuencias
def bandpass_boost(data, fs, f_low, f_high, gain_db=10):
    sos = butter(2, [f_low, f_high], btype='band', fs=fs, output='sos')
    boosted = sosfilt(sos, data)
    factor = 10**(gain_db/20)
    return boosted * factor

#Elimina la frecuencia que causa tinnitus para no sobreestimular
def notch_filter(data, fs, freq, q=30):
    bw = freq / q
    f1, f2 = freq - bw / 2, freq + bw / 2
    sos = butter(2, [f1, f2], btype='bandstop', fs=fs, output='sos')
    return sosfilt(sos, data)

# Boton generacion MTRS
if st.button("🎶 Generar sonido terapéutico"):
    low_band, high_band = get_mtrs_bands(freq)

    processed = notch_filter(samples, fs, freq)
    processed += bandpass_boost(samples, fs, *low_band)
    processed += bandpass_boost(samples, fs, *high_band)

    processed = processed / np.max(np.abs(processed)) * 32767
    processed = processed.astype(np.int16)

    output_filename = "MTRS_therapy.wav"
    with wave.open(output_filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(processed.tobytes())

    st.success("✅ Audio MTRS generado correctamente.")
    st.audio(output_filename, format="audio/wav")

    with open(output_filename, "rb") as f:
        st.download_button("⬇️ Descargar audio", f, file_name=output_filename, mime="audio/wav")

    os.remove(output_filename)

