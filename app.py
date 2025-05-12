import streamlit as st
import torch
import librosa
import librosa.display
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from models.dnn_model import DNNSpeechEnhancer

# --- Fungsi Plot Perbandingan ---
def plot_comparison(original, enhanced, sr):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # 1. Waveform Original vs Enhanced
    librosa.display.waveshow(original, sr=sr, ax=ax1, color='blue', alpha=0.6, label='Original')
    librosa.display.waveshow(enhanced, sr=sr, ax=ax1, color='red', alpha=0.5, label='Enhanced')
    ax1.set_title('Waveform Comparison')
    ax1.legend()

    # 2. Original Spectrogram
    S_orig = librosa.amplitude_to_db(np.abs(librosa.stft(original)), ref=np.max)
    img_orig = librosa.display.specshow(S_orig, sr=sr, ax=ax2, y_axis='log', x_axis='time')
    ax2.set_title('Original Spectrogram')
    fig.colorbar(img_orig, format='%+2.0f dB', ax=ax2)

    # 3. Enhanced Spectrogram
    S_enh = librosa.amplitude_to_db(np.abs(librosa.stft(enhanced)), ref=np.max)
    img_enh = librosa.display.specshow(S_enh, sr=sr, ax=ax3, y_axis='log', x_axis='time')
    ax3.set_title('Enhanced Spectrogram')
    fig.colorbar(img_enh, format='%+2.0f dB', ax=ax3)

    plt.tight_layout()
    st.pyplot(fig)

# --- Judul Aplikasi ---
st.title("🎤 Speech Enhancement Dashboard")
st.markdown("""
Uji model peningkat kualitas suara pada audio baru
""")

# --- Sidebar Upload ---
st.sidebar.header("Pengaturan")
audio_file = st.sidebar.file_uploader(
    "Upload file audio (format WAV)", 
    type=["wav"],
    accept_multiple_files=False
)

# --- Load Model ---
@st.cache_resource
def load_model():
    model = DNNSpeechEnhancer(**config['model_config'])
    checkpoint = torch.load("saved_models/20250427_195123/best_model.pth", map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

model = load_model()

# --- Main Area ---
if audio_file is not None:
    st.header("Input Audio")
    audio_bytes = audio_file.read()
    
    # Simpan file sementara
    temp_input = "temp_input.wav"
    with open(temp_input, "wb") as f:
        f.write(audio_bytes)
    
    # Load audio
    original_audio, sr = librosa.load(temp_input, sr=config['data_config']['sample_rate'])
    
    # Player
    st.audio(audio_bytes, format='audio/wav')
    
    # Waveform
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(original_audio, sr=sr, ax=ax)
    ax.set_title("Waveform Audio Asli")
    ax.set_xlabel("Waktu (s)")
    ax.set_ylabel("Amplitudo")
    st.pyplot(fig)
    
    # Tombol Enhancer
    if st.button("🎚️ Enhance Audio"):
        with st.spinner("Memproses audio..."):
            # Preprocessing
            if len(original_audio) < config['model_config']['input_size']:
                padded = np.pad(original_audio, (0, config['model_config']['input_size'] - len(original_audio)))
            else:
                padded = original_audio[:config['model_config']['input_size']]
            
            # Tensor
            audio_tensor = torch.FloatTensor(padded).unsqueeze(0)
            
            # Inference
            with torch.no_grad():
                enhanced = model(audio_tensor)
            enhanced_audio = enhanced.squeeze().numpy()

            temp_output = "temp_output.wav"
            sf.write(temp_output, enhanced_audio, sr)
            
            st.success("✅ Enhancement selesai!")
            
            # --- Hasil Audio ---
            st.header("Hasil Enhanced Audio")
            st.audio(temp_output, format='audio/wav')
            
            # Visualisasi Perbandingan
            st.header("Visualisasi Perbandingan")
            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            librosa.display.waveshow(original_audio, sr=sr, ax=ax1)
            ax1.set_title("Audio Asli")
            librosa.display.waveshow(enhanced_audio, sr=sr, ax=ax2)
            ax2.set_title("Audio Hasil Enhancement")
            plt.tight_layout()
            st.pyplot(fig2)
            
            # Spektrogram & Waveform Comparison
            st.header("Perbandingan Waveform dan Spektrogram")
            plot_comparison(original_audio, enhanced_audio, sr)

            # Tombol Download
            with open(temp_output, "rb") as f:
                btn = st.download_button(
                    label="⬇️ Download Enhanced Audio",
                    data=f,
                    file_name="enhanced_output.wav",
                    mime="audio/wav"
                )

else:
    st.info("Silakan upload file audio WAV melalui sidebar di sebelah kiri.")

# --- Footer ---
st.markdown("---")
st.markdown("""
**Tips Penggunaan:**
1. Upload file audio berformat WAV (16kHz recommended)
2. Klik tombol 'Enhance Audio' untuk memproses
3. Bandingkan hasil secara visual dan auditori
4. Download hasil enhancement jika memuaskan
""")
