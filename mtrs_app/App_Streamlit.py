import streamlit as st
import numpy as np
from pydub import AudioSegment
from scipy.signal import butter, sosfilt
import wave
import os

# ---------- Configuración inicial ----------
st.set_page_config(page_title="MTRS Generator", layout="centered")
st.title("🎧 MTRS Sound Therapy Generator")
st.write("Selecciona las características del tinnitus para generar un audio terapéutico personalizado.")

# ---------- Entrada del usuario ----------
freq = st.selectbox("Frecuencia del tinnitus (Hz)", [250, 500, 1000, 2000, 4000, 6000, 8000])
db = st.slider("Volumen percibido (dB HL)", 0, 80, 40)

# ---------- Cargar audio base ----------
audio = AudioSegment.from_file("ONDAS_DELTA_binaural-sound.wav", format="wav")
samples = np.array(audio.get_array_of_samples()).astype(np.float32)
fs = audio.frame_rate

# ---------- Procesamiento MTRS ----------
def get_mtrs_bands(center_freq):
    band_width = center_freq * 0.26
    lower_band = (center_freq - band_width, center_freq - band_width / 2)
    upper_band = (center_freq + band_width / 2, center_freq + band_width)
    return lower_band, upper_band

def bandpass_boost(data, fs, f_low, f_high, gain_db=10):
    sos = butter(2, [f_low, f_high], btype='band', fs=fs, output='sos')
    boosted = sosfilt(sos, data)
    factor = 10**(gain_db/20)
    return boosted * factor

def notch_filter(data, fs, freq, q=30):
    bw = freq / q
    f1, f2 = freq - bw / 2, freq + bw / 2
    sos = butter(2, [f1, f2], btype='bandstop', fs=fs, output='sos')
    return sosfilt(sos, data)

# ---------- Botón de generación ----------
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

    st.success("Audio generado correctamente!")
    st.audio(output_filename, format="audio/wav")

    with open(output_filename, "rb") as f:
        st.download_button("⬇️ Descargar audio", f, file_name=output_filename, mime="audio/wav")

    os.remove(output_filename)
