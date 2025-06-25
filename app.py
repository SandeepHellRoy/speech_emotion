import streamlit as st
import torch
import torchaudio
import librosa
import numpy as np
from model import EmotionCNNBiLSTMAttention  # your model class

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
MODEL_PATH = "./best_model.pt"

# Load model
@st.cache_resource
def load_model():
    model = EmotionCNNBiLSTMAttention(n_classes=len(CLASS_NAMES))
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    return model

model = load_model()

# Audio preprocessing
def preprocess_audio(file, sr=16000, max_len=300):
    y, _ = librosa.load(file, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    logmel = librosa.power_to_db(mel, ref=np.max)

    if logmel.shape[1] < max_len:
        pad = max_len - logmel.shape[1]
        logmel = np.pad(logmel, ((0, 0), (0, pad)), mode='constant')
    else:
        logmel = logmel[:, :max_len]

    logmel_tensor = torch.tensor(logmel).unsqueeze(0).unsqueeze(0)  # [1, 1, 128, T]
    return logmel_tensor.to(device)

# Streamlit UI
st.title("🎤 Speech Emotion Recognition App")
st.write("Upload a `.wav` file and see the predicted emotion.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    with st.spinner("Analyzing emotion..."):
        input_tensor = preprocess_audio(uploaded_file)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            pred_emotion = CLASS_NAMES[pred_idx]

        st.success(f"🎯 **Predicted Emotion**: `{pred_emotion}`")
        st.bar_chart({CLASS_NAMES[i]: probs[i] for i in range(len(CLASS_NAMES))})
