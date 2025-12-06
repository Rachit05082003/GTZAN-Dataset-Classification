#!/usr/bin/env python3
"""
CNN + Temporal Attention for Music Genre Classification
Optimized implementation with data augmentation and enhanced features
"""

import os
import argparse
import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ==================== CONFIGURATION ====================

class Config:
    """Configuration parameters"""
    # Paths
    DATA_PATH = '/Users/narac0503/GIT/GTZAN Dataset Classification/GTZAN-Dataset-Classification/gtzan-classification/data/gtzan/genres_original'
    OUTPUT_DIR = "output"

    # Audio parameters
    SAMPLE_RATE = 22050
    DURATION = 30
    N_MFCC = 40
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128

    # Segmentation for attention
    NUM_SEGMENTS = 10
    SEGMENT_DURATION = 3

    # Model parameters
    NUM_CLASSES = 10
    GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']

    # Hyperparameters
    CNN_FILTERS = [32, 64, 128, 256]
    ATTENTION_DIM = 128
    DENSE_UNITS = 256
    DROPOUT = 0.5
    L2_REG = 0.001

    LEARNING_RATE = 0.0003
    BATCH_SIZE = 32
    EPOCHS = 100

    # Training options
    USE_AUGMENTATION = True
    AUGMENTATION_FACTOR = 4  # Generate 4 versions per file
    TEST_SIZE = 0.2
    RANDOM_SEED = 42

config = Config()

# Set seeds
np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)


# ==================== DATA AUGMENTATION ====================

def augment_audio(y, sr=22050):
    """
    Apply random audio augmentation.
    Returns augmented audio of same length.
    """
    augmentations = []

    # Original
    augmentations.append(y)

    # Time stretch (±5%)
    if np.random.rand() > 0.5:
        rate = np.random.uniform(0.95, 1.05)
        y_stretched = librosa.effects.time_stretch(y, rate=rate)
        # Ensure same length
        if len(y_stretched) > len(y):
            y_stretched = y_stretched[:len(y)]
        else:
            y_stretched = np.pad(y_stretched, (0, len(y) - len(y_stretched)))
        augmentations.append(y_stretched)

    # Pitch shift (±2 semitones)
    if np.random.rand() > 0.5:
        n_steps = np.random.randint(-2, 3)
        if n_steps != 0:
            y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
            augmentations.append(y_shifted)

    # Add white noise
    if np.random.rand() > 0.5:
        noise_level = np.random.uniform(0.002, 0.005)
        noise = np.random.normal(0, noise_level, len(y))
        y_noise = y + noise
        augmentations.append(y_noise)

    return augmentations


# ==================== FEATURE EXTRACTION ====================

def extract_enhanced_mfcc(y, sr=22050):
    """
    Extract enhanced MFCC features with deltas and delta-deltas.
    Returns: (n_mfcc * 3, time_frames)
    """
    # Pre-emphasis filter
    pre_emphasis = 0.97
    y_pre = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    # Extract MFCC
    mfcc = librosa.feature.mfcc(
        y=y_pre, sr=sr,
        n_mfcc=config.N_MFCC,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        fmin=20,
        fmax=8000
    )

    # Add deltas
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Stack all features
    enhanced = np.vstack([mfcc, delta, delta2])

    return enhanced


def extract_segmented_features(audio_path, augment=False):
    """
    Load audio, optionally augment, and extract segmented MFCC features.
    Returns: list of (num_segments, n_features, time_frames) arrays
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE, duration=config.DURATION)

        # Ensure exact length
        target_len = config.SAMPLE_RATE * config.DURATION
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        # Generate augmentations if requested
        if augment:
            audio_versions = augment_audio(y, sr)[:config.AUGMENTATION_FACTOR]
        else:
            audio_versions = [y]

        all_features = []

        for audio in audio_versions:
            # Split into segments
            segment_length = len(audio) // config.NUM_SEGMENTS
            segments = []

            for i in range(config.NUM_SEGMENTS):
                start = i * segment_length
                end = start + segment_length
                segment_audio = audio[start:end]

                # Extract enhanced MFCC for this segment
                mfcc = extract_enhanced_mfcc(segment_audio, sr)
                segments.append(mfcc)

            segments = np.array(segments)
            all_features.append(segments)

        return all_features

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def load_dataset(data_path, augment=False):
    """
    Load entire GTZAN dataset with optional augmentation.
    Returns: X (features), y (labels)
    """
    X, y = [], []

    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)

    for genre in config.GENRES:
        genre_path = os.path.join(data_path, genre)
        if not os.path.exists(genre_path):
            print(f"Warning: {genre} folder not found")
            continue

        files = sorted([f for f in os.listdir(genre_path) if f.endswith('.wav')])
        print(f"\n{genre.upper()}: {len(files)} files")

        loaded = 0
        for f in files:
            fpath = os.path.join(genre_path, f)
            features_list = extract_segmented_features(fpath, augment=augment)

            if features_list is not None:
                for features in features_list:
                    X.append(features)
                    y.append(genre)
                loaded += 1

                if loaded % 10 == 0:
                    print(f"  Processed: {loaded}/{len(files)}", end='\r')

        print(f"  ✓ Loaded: {loaded} files → {len(features_list) * loaded} samples")

    X = np.array(X)
    y = np.array(y)

    print(f"\n{'='*60}")
    print(f"Total samples: {len(X)}")
    print(f"Feature shape: {X.shape}")
    print(f"{'='*60}\n")

    return X, y


# ==================== ATTENTION MECHANISM ====================

class TemporalAttention(layers.Layer):
    """
    Temporal Attention layer.
    Learns importance weights for temporal segments.
    """

    def __init__(self, attention_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = attention_dim

    def build(self, input_shape):
        feature_dim = input_shape[-1]

        self.W = self.add_weight(
            shape=(feature_dim, self.attention_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_W'
        )
        self.b = self.add_weight(
            shape=(self.attention_dim,),
            initializer='zeros',
            trainable=True,
            name='attention_b'
        )
        self.u = self.add_weight(
            shape=(self.attention_dim, 1),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_u'
        )
        super().build(input_shape)

    def call(self, x):
        # Compute attention scores
        score = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        score = tf.tensordot(score, self.u, axes=1)
        score = tf.squeeze(score, -1)

        # Softmax weights
        attention_weights = tf.nn.softmax(score, axis=-1)

        # Weighted sum
        weighted = tf.expand_dims(attention_weights, -1) * x
        output = tf.reduce_sum(weighted, axis=1)

        return output

    def get_config(self):
        config = super().get_config()
        config['attention_dim'] = self.attention_dim
        return config


# ==================== MODEL ARCHITECTURE ====================

def build_segment_cnn(input_shape):
    """Build CNN to process each segment."""
    inputs = layers.Input(shape=input_shape)
    x = inputs

    for i, filters in enumerate(config.CNN_FILTERS):
        x = layers.Conv2D(
            filters, (3, 3), padding='same',
            kernel_regularizer=regularizers.l2(config.L2_REG),
            name=f'conv_{i+1}'
        )(x)
        x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
        x = layers.Activation('relu', name=f'relu_{i+1}')(x)
        x = layers.MaxPooling2D((2, 2), name=f'pool_{i+1}')(x)
        x = layers.Dropout(0.25, name=f'dropout_{i+1}')(x)

    x = layers.GlobalAveragePooling2D(name='gap')(x)

    return Model(inputs, x, name='segment_cnn')


def build_cnn_attention_model(input_shape):
    """
    Build complete CNN + Attention model.
    """
    num_segments = input_shape[0]
    segment_shape = input_shape[1:]

    # Input
    inputs = layers.Input(shape=input_shape, name='input')

    # Segment CNN
    segment_cnn = build_segment_cnn(segment_shape)

    # Apply CNN to each segment
    features = layers.TimeDistributed(segment_cnn, name='time_distributed')(inputs)

    # Temporal Attention
    attended = TemporalAttention(config.ATTENTION_DIM, name='attention')(features)

    # Classification head
    x = layers.Dense(
        config.DENSE_UNITS,
        kernel_regularizer=regularizers.l2(config.L2_REG),
        name='dense_1'
    )(attended)
    x = layers.BatchNormalization(name='bn_dense_1')(x)
    x = layers.Activation('relu', name='relu_dense')(x)
    x = layers.Dropout(config.DROPOUT, name='dropout_dense')(x)

    x = layers.Dense(
        config.DENSE_UNITS // 2,
        kernel_regularizer=regularizers.l2(config.L2_REG),
        name='dense_2'
    )(x)
    x = layers.BatchNormalization(name='bn_dense_2')(x)
    x = layers.Activation('relu', name='relu_dense_2')(x)
    x = layers.Dropout(config.DROPOUT, name='dropout_dense_2')(x)

    outputs = layers.Dense(config.NUM_CLASSES, activation='softmax', name='output')(x)

    model = Model(inputs, outputs, name='CNN_Attention')
    return model


# ==================== TRAINING ====================

def train_model(X_train, y_train, X_val, y_val, output_dir):
    """Train the model with proper callbacks."""

    print("\n" + "="*60)
    print("BUILDING MODEL")
    print("="*60)

    input_shape = X_train.shape[1:]
    model = build_cnn_attention_model(input_shape)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(output_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(output_dir, 'logs'),
            histogram_freq=1
        )
    ]

    print("\n" + "="*60)
    print("TRAINING")
    print("="*60 + "\n")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    return model, history


# ==================== EVALUATION ====================

def evaluate_model(model, X_test, y_test, output_dir):
    """Evaluate model and generate reports."""

    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    # Predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Accuracy
    acc = accuracy_score(y_test, y_pred_labels)
    print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # Classification report
    print("\nClassification Report:")
    report = classification_report(
        y_test, y_pred_labels,
        target_names=config.GENRES,
        digits=3
    )
    print(report)

    # Save report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.GENRES, yticklabels=config.GENRES)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(f'Confusion Matrix (Acc: {acc:.2%})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()

    return acc, report


def plot_training_history(history, output_dir):
    """Plot and save training curves."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(history.history['loss'], label='Train', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()

    best_val = max(history.history['val_accuracy'])
    print(f"\nBest Validation Accuracy: {best_val:.4f} ({best_val*100:.2f}%)")


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description='CNN + Attention for Music Genre Classification')
    parser.add_argument('--data_path', type=str, default=config.DATA_PATH,
                        help='Path to GTZAN dataset')
    parser.add_argument('--output_dir', type=str, default=config.OUTPUT_DIR,
                        help='Output directory for results')
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                        help='Learning rate')

    args = parser.parse_args()

    # Update config
    config.DATA_PATH = args.data_path
    config.OUTPUT_DIR = args.output_dir
    config.USE_AUGMENTATION = not args.no_augment
    config.BATCH_SIZE = args.batch_size
    config.EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.OUTPUT_DIR, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("CNN + TEMPORAL ATTENTION - MUSIC GENRE CLASSIFICATION")
    print("="*60)
    print(f"Data path: {config.DATA_PATH}")
    print(f"Output dir: {output_dir}")
    print(f"Augmentation: {config.USE_AUGMENTATION}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print("="*60)

    # Check GPU
    print(f"\nGPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

    # Load data
    X, y = load_dataset(config.DATA_PATH, augment=config.USE_AUGMENTATION)

    # Add channel dimension
    X = X[..., np.newaxis]

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=config.TEST_SIZE,
        stratify=y_enc,
        random_state=config.RANDOM_SEED
    )

    # Normalize
    mean = X_train.mean()
    std = X_train.std()
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")

    # Save normalization params
    np.savez(os.path.join(output_dir, 'normalization.npz'), mean=mean, std=std)

    # Train model
    model, history = train_model(X_train, y_train, X_test, y_test, output_dir)

    # Plot training history
    plot_training_history(history, output_dir)

    # Evaluate
    acc, report = evaluate_model(model, X_test, y_test, output_dir)

    # Save final model
    model.save(os.path.join(output_dir, 'final_model.h5'))

    print(f"\n{'='*60}")
    print(f"RESULTS SAVED TO: {output_dir}")
    print(f"Final Test Accuracy: {acc:.2%}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
