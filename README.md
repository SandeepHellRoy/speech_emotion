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


## 🏋️‍♂️ Training Details

- **Preprocessing:**
  - Audio is resampled to **16 kHz** using `librosa.load`.
  - Extracted features are **Log-Mel spectrograms** with:
    - `n_mels = 128`
    - Temporal dimension padded/truncated to `max_len = 300`
  - Normalization: Each audio signal is peak-normalized before feature extraction.

- **Data Augmentation:**
  - Random **time-stretching**
  - **Pitch shifting** within ±2 semitones
  - Additive **Gaussian noise**
  - These augmentations help generalize the model on a limited dataset.

- **Handling Class Imbalance:**
  - **Class weights** were computed using `sklearn.utils.compute_class_weight` and used in the cross-entropy loss function to give more weight to rare emotion classes.

- **Model Architecture:**
  - CNN layers extract spatial patterns from spectrograms.
  - BiLSTM captures temporal dependencies.
  - Attention mechanism focuses on important time frames.
  - Dropout used in CNN and LSTM layers to prevent overfitting.

- **Training Parameters:**
  - Optimizer: `Adam`
  - Learning rate: `1e-3`
  - Batch size: `16`
  - Early stopping: Patience of `10` epochs without improvement
  - Maximum epochs: `100`
  - Best model checkpoint saved based on validation accuracy

---

## 📊 Evaluation Metrics

- **Overall Accuracy**: Measures the percentage of correctly predicted samples.
- **Class-wise Accuracy**: Accuracy calculated per emotion class to reveal imbalance performance.
- **F1 Score**: Harmonic mean of precision and recall for each class.
- **Confusion Matrix**: Displays predicted vs actual labels to visualize common misclassifications.

All metrics are logged during evaluation and used to analyze the model's behavior across different emotions.

