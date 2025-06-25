import zipfile
import os
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
from tqdm import tqdm
import warnings
import h5py

warnings.filterwarnings(action="ignore")

dataset_link = "https://zenodo.org/records/1188976#.XCx-tc9KhQI"
# dataset_folder = "/content/drive/MyDrive/Mars"
extraction_path = "./dataset"

os.makedirs(extraction_path, exist_ok=True)

# zip_files = ['Audio_Speech_Actors_01-24.zip', 'Audio_Song_Actors_01-24.zip']
# for z in zip_files:
#     with zipfile.ZipFile(os.path.join(dataset_folder, z), 'r') as zip_ref:
#         zip_ref.extractall(extraction_path)

# ✅ Confirm number of .wav files extracted
total_files = sum([len(files) for r, d, files in os.walk(extraction_path)])
print(f"Total .wav files extracted: {total_files}")  # Should be ~1440

emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Extract emotion label from filename
def extract_emotion_label(filename):
    parts = filename.split('-')
    emotion_code = parts[2]
    return emotion_dict.get(emotion_code)

def extract_features_full(file_path, n_mfcc=40, max_pad_len=862):
    try:
        y, sr = librosa.load(file_path, sr=None)
        if y is None or len(y) == 0:
            raise ValueError("Empty audio signal")

        features = []

        # 1️⃣ MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1)
        features.extend(mfcc_mean)
        features.extend(mfcc_delta_mean)
        features.extend(mfcc_delta2_mean)

        # 2️⃣ Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1))

        # 3️⃣ Spectral Contrast
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features.extend(np.mean(spec_contrast, axis=1))

        # 4️⃣ Tonnetz (harmonic features)
        y_harmonic = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
        features.extend(np.mean(tonnetz, axis=1))

        # 5️⃣ Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zcr))

        # 6️⃣ Root Mean Square Energy
        rmse = librosa.feature.rms(y=y)
        features.append(np.mean(rmse))

        # 7️⃣ Pitch (Fundamental Frequency)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch = pitches[magnitudes > np.median(magnitudes)]
        if pitch.size > 0:
            features.append(np.mean(pitch))
            features.append(np.std(pitch))
        else:
            features.extend([0, 0])

        # 8️⃣ Harmonic-to-Noise Ratio (HNR)
        # Approximation using SNR (since librosa doesn’t provide HNR directly)
        S, phase = librosa.magphase(librosa.stft(y))
        harmonic = librosa.effects.harmonic(y)
        noise = librosa.effects.percussive(y)
        hnr = np.mean(harmonic*2) / (np.mean(noise*2) + 1e-6)
        features.append(hnr)

        return np.array(features)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Process single file → (features, label)
def process_file(file_path, file):
    try:
        label = extract_emotion_label(file)
        features = extract_features_full(file_path)
        if features is not None and label is not None:
            return features, label
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
    return None


# Full dataset processing with parallelism
def build_feature_dataset_parallel(directory, n_jobs=1):
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                file_list.append((file_path, file))

    results = Parallel(n_jobs=n_jobs)(
    delayed(process_file)(fp, f) for fp, f in tqdm(file_list)
    )


    features = [r[0] for r in results if r]
    labels = [r[1] for r in results if r]
    return np.array(features), np.array(labels)

def build_feature_dataset_hdf5(directory, hdf5_file='features_labels.h5'):
    # First count valid files (for preallocation)
    print("Counting valid .wav files...")
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                file_list.append((file_path, file))

    total = len(file_list)
    print(f"Total valid .wav files: {total}")

    # Setup HDF5 file
    with h5py.File(hdf5_file, 'w') as f:
        max_feature_len = 0
        sample_shape = extract_features_full(file_list[0][0]).shape[0]
        X_ds = f.create_dataset('X', shape=(total, sample_shape), dtype=np.float32)
        y_ds = f.create_dataset('y', shape=(total,), dtype=h5py.string_dtype())

        idx = 0
        for file_path, file in tqdm(file_list):
            result = process_file(file_path, file)
            if result:
                features, label = result
                X_ds[idx] = features
                y_ds[idx] = label
                idx += 1

        # Resize datasets if some files were skipped
        X_ds.resize((idx, sample_shape))
        y_ds.resize((idx,))
        print(f"Saved {idx} samples to {hdf5_file}")



# Save features and labels
def save_dataset(X, y, feature_file='X_features.npy', label_file='y_labels.npy'):
    np.save(feature_file, X)
    np.save(label_file, y)
    print(f"Saved features to {feature_file} and labels to {label_file}")

# Load features and labels
def load_dataset(feature_file='X_features.npy', label_file='y_labels.npy'):
    X = np.load(feature_file)
    y = np.load(label_file)
    return X, y

def load_dataset_hdf5(hdf5_file='features_labels.h5'):
    with h5py.File(hdf5_file, 'r') as f:
        X = np.array(f['X'])
        y = np.array(f['y']).astype(str)  # decode string labels
    return X, y


X, y = build_feature_dataset_hdf5(extraction_path)
save_dataset(X, y)

X, y = load_dataset()
print("Feature shape:", X.shape)
print("Labels shape:", y.shape)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print("Train size:", X_train.shape[0])
print("Validation size:", X_val.shape[0])

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Compute class weights for imbalance
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))

# Convert class_weight dict to list (XGBoost uses scale_pos_weight → we’ll pass weights manually)
sample_weights = np.array([class_weight_dict[label] for label in y_train])

# Build model
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(classes),
    n_estimators=400,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,        # L1 regularization → prevents overfitting
    reg_lambda=1.0,       # L2 regularization → prevents overfitting
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)

# Train with sample weights
model.fit(X_train, y_train, sample_weight=sample_weights)

y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Code to find nunique in dataset:
# prompt: find no of no of unique values

unique_labels, counts = np.unique(y, return_counts=True)

print("Unique labels:", unique_labels)
print("Counts of unique labels:", counts)

plt.figure(figsize=(10, 6))
sns.barplot(x=unique_labels, y=counts, palette="viridis")
plt.title("Distribution of Emotions in the Dataset")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.show()
# prompt: find no of no of unique values

unique_labels, counts = np.unique(y, return_counts=True)

print("Unique labels:", unique_labels)
print("Counts of unique labels:", counts)

plt.figure(figsize=(10, 6))
sns.barplot(x=unique_labels, y=counts, palette="viridis")
plt.title("Distribution of Emotions in the Dataset")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.show()