# 🎧 Speech Emotion Recognition using CNN-BiLSTM-Attention

This project focuses on classifying human emotions from audio recordings using a deep learning model based on a CNN-BiLSTM architecture with attention. It uses log-mel spectrograms for input representation and handles class imbalance using feature weighting. The project includes training, evaluation, and inference scripts.

---

## 📌 Project Description

Speech Emotion Recognition (SER) aims to identify the underlying emotion in human speech. Applications include call center automation, affective computing, human-computer interaction, and mental health analysis.

This project builds a robust SER system using the **RAVDESS** dataset and a deep neural network architecture. The network combines convolutional and recurrent layers with an attention mechanism to capture both spatial and temporal emotional patterns in speech.

---

## 📂 Dataset

We use the [RAVDESS](https://zenodo.org/record/1188976) dataset — an English audio-visual dataset consisting of 24 professional actors speaking with various emotions.

### Emotion Labels:
| ID | Emotion     |
|----|-------------|
| 01 | Neutral     |
| 02 | Calm        |
| 03 | Happy       |
| 04 | Sad         |
| 05 | Angry       |
| 06 | Fearful     |
| 07 | Disgust     |
| 08 | Surprised   |

Each filename contains the emotion ID at position 3 (e.g. `03-01-01-01-01-01-06.wav` → `fearful`).

---

## ⚙️ Preprocessing

- **Sampling Rate**: All audio is resampled to 16,000 Hz.
- **Feature Extraction**: Log-Mel spectrograms (`128` mel bands, time length padded/truncated to `T=300`).
- **Normalization**: Audio is normalized to unit scale.
- **Augmentation** *(training only)*:
  - Time-stretching
  - Pitch shifting
  - Gaussian noise

---

## 🧠 Model Architecture

### `EmotionCNNBiLSTMAttention`

```python
Input: [B, 1, 128, T]

CNN:
├── Conv2d(1 → 32) → ReLU → BatchNorm2d → Dropout2d(0.2) → MaxPool2d
├── Conv2d(32 → 64) → ReLU → BatchNorm2d → Dropout2d(0.2) → MaxPool2d

→ Reshape → [B, T, Features=64*32]

BiLSTM:
├── BiLSTM(input=64*32, hidden=128, bidirectional=True)
├── Dropout(0.2)

Attention:
├── Linear(256 → 1) + Softmax → Weighted sum

FC:
├── Linear(256 → n_classes)
