import os
import torch
import librosa
import numpy as np
import argparse
import pandas as pd
from model import EmotionCNNBiLSTMAttention  # Make sure this import path is correct

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/best_model.pt"
CLASS_NAMES = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
SR = 16000
MAX_LEN = 300

# Load model
def load_model():
    model = EmotionCNNBiLSTMAttention(n_classes=len(CLASS_NAMES))
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(DEVICE)
    return model

# Preprocessing
def extract_logmel(path, sr=SR, max_len=MAX_LEN):
    y, _ = librosa.load(path, sr=sr)
    y = y / np.max(np.abs(y))  # normalize
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    logmel = librosa.power_to_db(mel, ref=np.max)

    if logmel.shape[1] < max_len:
        pad = max_len - logmel.shape[1]
        logmel = np.pad(logmel, ((0, 0), (0, pad)), mode='constant')
    else:
        logmel = logmel[:, :max_len]

    return torch.tensor(logmel).unsqueeze(0).unsqueeze(0).float().to(DEVICE)  # [1, 1, 128, T]

# Inference
def predict(model, path):
    input_tensor = extract_logmel(path)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        return CLASS_NAMES[pred_idx], probs

# Run on folder
def run_on_folder(folder, model):
    results = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(".wav"):
            path = os.path.join(folder, fname)
            try:
                label, _ = predict(model, path)
                print(f"{fname} -> {label}")
                results.append({"file": fname, "predicted_label": label})
            except Exception as e:
                print(f"Error processing {fname}: {e}")
    df = pd.DataFrame(results)
    df.to_csv("inference_results.csv", index=False)
    print("✅ Results saved to `inference_results.csv`")

# Run on single file
def run_on_file(file, model):
    label, _ = predict(model, file)
    print(f"{os.path.basename(file)} -> {label}")

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion recognition inference script")
    parser.add_argument("--file", type=str, help="Path to a single .wav file")
    parser.add_argument("--folder", type=str, help="Path to folder containing .wav files")

    args = parser.parse_args()
    model = load_model()

    if args.file:
        run_on_file(args.file, model)
    elif args.folder:
        run_on_folder(args.folder, model)
    else:
        print("❌ Please provide --file or --folder argument.")
