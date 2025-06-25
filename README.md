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

```

## 📊 Evaluation Metrics

- **Overall Accuracy**: Measures the percentage of correctly predicted samples.
- **Class-wise Accuracy**: Accuracy calculated per emotion class to reveal imbalance performance.
- **F1 Score**: Harmonic mean of precision and recall for each class.
- **Confusion Matrix**: Displays predicted vs actual labels to visualize common misclassifications.

All metrics are logged during evaluation and used to analyze the model's behavior across different emotions.

## ✅ Evaluation Results

- **Overall Accuracy**: `73.32%`
- **Weighted F1 Score**: `73.25%`

### 📊 Confusion Matrix

```
[[63  0  6  0  4  0  0  2]
 [ 0 62  2  0  5  4  2  0]
 [ 6  1 27  2  1  0  1  1]
 [ 4  1  0 46  2  0 16  6]
 [ 3  1  1  9 50  3  6  2]
 [ 0  2  1  0  0 32  3  0]
 [ 2  6  5  6  4  3 49  0]
 [ 0  0  1  1  2  1  3 31]]
```

### 🎯 Accuracy Per Class

- **Angry**: 84.00%
- **Calm**: 82.67%
- **Disgust**: 69.23%
- **Fearful**: 61.33%
- **Happy**: 66.67%
- **Neutral**: 84.21%
- **Sad**: 65.33%
- **Surprised**: 79.49%

---

### 📋 Classification Report

```
              precision    recall  f1-score   support

       angry       0.81      0.84      0.82        75
        calm       0.85      0.83      0.84        75
     disgust       0.63      0.69      0.66        39
     fearful       0.72      0.61      0.66        75
       happy       0.74      0.67      0.70        75
     neutral       0.74      0.84      0.79        38
         sad       0.61      0.65      0.63        75
   surprised       0.74      0.79      0.77        39

    accuracy                           0.73       491
   macro avg       0.73      0.74      0.73       491
weighted avg       0.73      0.73      0.73       491
```
