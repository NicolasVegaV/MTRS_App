import streamlit as st
import numpy as np
from pydub import AudioSegment
from scipy.signal import butter, sosfilt
import wave
import os
import tempfile
from scipy.io.wavfile import write as wav_write

# Configuración inicial
st.set_page_config(page_title="MTRS Generator", layout="centered")
st.title("🎧 MTRS Sound Therapy Generator")
st.write("Select the tinnitus characteristics to generate a personalised therapeutic audio.")

# Funciones auxiliares

def dbhl_to_amplitude(db_hl):
    return 10 ** ((db_hl - 80) / 20)

def generate_tone(frequency, duration=1.0, fs=44100, db_hl=40):
    amplitude = dbhl_to_amplitude(db_hl)
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    tone = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    return tone, fs

def get_mtrs_bands(center_freq):
    band_width = center_freq * 0.26
    lower_band = (center_freq - band_width, center_freq - band_width / 2)
    upper_band = (center_freq + band_width / 2, center_freq + band_width)
    return lower_band, upper_band

def bandpass_boost(data, fs, f_low, f_high, gain_db=10):
    sos = butter(2, [f_low, f_high], btype='band', fs=fs, output='sos')
    boosted = sosfilt(sos, data)
    factor = 10 ** (gain_db / 20)
    return boosted * factor

def notch_filter_range(data, fs, f_low, f_high):
    sos = butter(2, [f_low, f_high], btype='bandstop', fs=fs, output='sos')
    return sosfilt(sos, data)

# Entrada de volumen
db = st.slider("Perceived volume (dB HL)", 0, 80, 40)

# Identificación asistida del tinnitus
st.markdown("## Assisted identification of tinnitus")

# Inicialización del estado
if 'min_freq' not in st.session_state:
    st.session_state.min_freq = 250
    st.session_state.max_freq = 8000
    st.session_state.step = 250
    st.session_state.selected = False

if 'button_pressed' not in st.session_state:
    st.session_state.button_pressed = None

# Procesamiento del botón presionado
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Lower"):
        st.session_state.button_pressed = 'lower'

with col2:
    if st.button("Higher"):
        st.session_state.button_pressed = 'higher'

with col3:
    if st.button("This is my tinnitus"):
        st.session_state.selected = True
        st.session_state.button_pressed = None

# Aplicar el cambio de frecuencia
if st.session_state.button_pressed == 'lower':
    st.session_state.max_freq = st.session_state.get('current_freq', (st.session_state.min_freq + st.session_state.max_freq) // 2)
    st.session_state.button_pressed = None

elif st.session_state.button_pressed == 'higher':
    st.session_state.min_freq = st.session_state.get('current_freq', (st.session_state.min_freq + st.session_state.max_freq) // 2)
    st.session_state.button_pressed = None

# Calcular frecuencia actual con los valores actualizados
st.session_state.current_freq = (st.session_state.min_freq + st.session_state.max_freq) // 2

# Mostrar frecuencia actual
st.write(f"Current frequency: **{st.session_state.current_freq} Hz**")
tone, fs = generate_tone(st.session_state.current_freq, duration=1.0, db_hl=db)
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
wav_write(tmp.name, fs, (tone * 32767).astype(np.int16))
st.audio(tmp.name, format="audio/wav")

# Mostrar resultado estimado
if st.session_state.selected:
    f1 = st.session_state.min_freq
    f2 = st.session_state.max_freq
    st.success(f"Estimated range of tinnitus: **{f1} Hz – {f2} Hz**")
    freq_center = (f1 + f2) / 2
else:
    st.stop()

# Cargar audio base
audio_path = os.path.join(os.path.dirname(__file__), "static", "sonido_mar_short.wav")
audio = AudioSegment.from_file(audio_path, format="wav")
samples = np.array(audio.get_array_of_samples()).astype(np.float32)
fs = audio.frame_rate

# Generar sonido terapéutico MTRS
if st.button("🎶 Generate therapeutic sound"):

    # Calcular bandas MTRS
    low_band, high_band = get_mtrs_bands(freq_center)

    # Aplicar procesamiento
    processed = notch_filter_range(samples, fs, f1, f2)
    processed += bandpass_boost(samples, fs, *low_band)
    processed += bandpass_boost(samples, fs, *high_band)

    # Normalizar y guardar
    processed = processed / np.max(np.abs(processed)) * 32767
    processed = processed.astype(np.int16)
    output_filename = "MTRS_therapy.wav"

    with wave.open(output_filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(processed.tobytes())

    st.success("✅ MTRS audio generated successfully.")
    st.audio(output_filename, format="audio/wav")

    st.markdown("### MTRS bands used")
    st.write(f" Reinforced bottom band: **{int(low_band[0])} Hz – {int(low_band[1])} Hz**")
    st.write(f" Reinforced upper band: **{int(high_band[0])} Hz – {int(high_band[1])} Hz**")
    st.write(f" Attenuated tinnitus frequency (notch): **{f1} Hz – {f2} Hz**")

    with open(output_filename, "rb") as f:
        st.download_button("⬇️ Download audio", f, file_name=output_filename, mime="audio/wav")

    os.remove(output_filename)
