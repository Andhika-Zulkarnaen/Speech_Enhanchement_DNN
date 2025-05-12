import streamlit as st
import torch
import librosa
import librosa.display
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from models.dnn_model import DNNSpeechEnhancer
from train import config
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d

# --- Fungsi Post-Processing ---
def highpass_filter(data, sr, cutoff=60, order=5):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def postprocess_audio(audio, sr):
    # High-pass filter
    filtered = highpass_filter(audio, sr)
    # Smooth transient clicks
    smoothed = gaussian_filter1d(filtered, sigma=0.4)
    # Normalisasi
    return smoothed / (np.max(np.abs(smoothed)) + 1e-7)

# --- Fungsi Plot ---
def plot_comparison(original, enhanced, sr):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    librosa.display.waveshow(original, sr=sr, ax=ax1, color='blue', alpha=0.6, label='Original')
    librosa.display.waveshow(enhanced, sr=sr, ax=ax1, color='red', alpha=0.5, label='Enhanced')
    ax1.set_title('Waveform Comparison')
    ax1.legend()

    S_orig = librosa.amplitude_to_db(np.abs(librosa.stft(original)), ref=np.max)
    img_orig = librosa.display.specshow(S_orig, sr=sr, ax=ax2, y_axis='log', x_axis='time')
    ax2.set_title('Original Spectrogram')
    fig.colorbar(img_orig, format='%+2.0f dB', ax=ax2)

    S_enh = librosa.amplitude_to_db(np.abs(librosa.stft(enhanced)), ref=np.max)
    img_enh = librosa.display.specshow(S_enh, sr=sr, ax=ax3, y_axis='log', x_axis='time')
    ax3.set_title('Enhanced Spectrogram')
    fig.colorbar(img_enh, format='%+2.0f dB', ax=ax3)

    plt.tight_layout()
    st.pyplot(fig)

# --- Judul Aplikasi ---
st.title("🎤 Speech Enhancement Dashboard")
st.markdown("Uji model peningkat kualitas suara pada audio baru")

# --- Sidebar ---
st.sidebar.header("Pengaturan")
audio_file = st.sidebar.file_uploader("Upload file audio (format WAV)", type=["wav"])

# --- Load Model ---
@st.cache_resource
def load_model():
    model = DNNSpeechEnhancer(**config['model_config'])
    checkpoint = torch.load("saved_models/20250427_195123/best_model.pth", map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

model = load_model()

# --- Main UI ---
if audio_file is not None:
    st.header("Input Audio")
    audio_bytes = audio_file.read()
    
    temp_input = "temp_input.wav"
    with open(temp_input, "wb") as f:
        f.write(audio_bytes)
    
    original_audio, sr = librosa.load(temp_input, sr=config['data_config']['sample_rate'])
    st.audio(audio_bytes, format='audio/wav')

    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(original_audio, sr=sr, ax=ax)
    ax.set_title("Waveform Audio Asli")
    st.pyplot(fig)
    
    if st.button("🎚️ Enhance Audio"):
        with st.spinner("Memproses audio..."):
            # Potong atau padding
            input_size = config['model_config']['input_size']
            if len(original_audio) < input_size:
                padded = np.pad(original_audio, (0, input_size - len(original_audio)))
            else:
                padded = original_audio[:input_size]
            
            # Inference
            audio_tensor = torch.FloatTensor(padded).unsqueeze(0)
            with torch.no_grad():
                enhanced = model(audio_tensor)
            enhanced_audio = enhanced.squeeze().numpy()

            # Post-Enhancement Processing
            import scipy.signal as sps

            # 1. Bandpass Filter (Hindari noise frekuensi rendah dan tinggi)
            def apply_bandpass(audio, sr, lowcut=80.0, highcut=7500.0):
              sos = sps.butter(10, [lowcut, highcut], btype='band', fs=sr, output='sos')
              return sps.sosfilt(sos, audio)

            # 2. Amplitude Noise Gate (hilangkan noise kecil < threshold)
            def noise_gate(audio, threshold=0.01):
              return np.where(np.abs(audio) < threshold, 0, audio)

            # 3. Normalisasi hasil
            def normalize_audio(audio):
              return audio / np.max(np.abs(audio))

            # Terapkan filtering
            enhanced_audio = apply_bandpass(enhanced_audio, sr)
            enhanced_audio = noise_gate(enhanced_audio, threshold=0.02)
            enhanced_audio = normalize_audio(enhanced_audio)

            # Post-processing
            enhanced_audio = postprocess_audio(enhanced_audio, sr)

            temp_output = "temp_output.wav"
            sf.write(temp_output, enhanced_audio, sr)

            st.success("✅ Enhancement selesai!")

            st.header("Hasil Enhanced Audio")
            st.audio(temp_output, format='audio/wav')

            # Visualisasi
            st.header("Visualisasi Perbandingan")
            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            librosa.display.waveshow(original_audio, sr=sr, ax=ax1)
            ax1.set_title("Audio Asli")
            librosa.display.waveshow(enhanced_audio, sr=sr, ax=ax2)
            ax2.set_title("Audio Hasil Enhancement")
            plt.tight_layout()
            st.pyplot(fig2)

            st.header("Perbandingan Waveform dan Spektrogram")
            plot_comparison(original_audio, enhanced_audio, sr)

            with open(temp_output, "rb") as f:
                st.download_button(
                    label="⬇️ Download Enhanced Audio",
                    data=f,
                    file_name="enhanced_output.wav",
                    mime="audio/wav"
                )
else:
    st.info("Silakan upload file audio WAV melalui sidebar di sebelah kiri.")

st.markdown("---")
st.markdown("""
**Tips Penggunaan:**
- Upload file audio WAV 16kHz
- Tekan tombol "Enhance Audio" untuk membersihkan
- Download hasil jika sudah sesuai
""")
