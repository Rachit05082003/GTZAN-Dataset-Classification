"""
GTZAN Genre Classification Web Demo

Run: python app.py
Open: http://localhost:5001
"""

import os
import uuid
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =============================================================================
# Configuration - MUST match training notebooks exactly
# =============================================================================
SAMPLE_RATE = 22050
DURATION = 30
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
NUM_SEGMENTS = 10
TARGET_LENGTH = 1291

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
UPLOAD_FOLDER = os.path.join(SCRIPT_DIR, 'uploads')

# Model paths - 3 models with normalization
MODEL_CONFIGS = {
    'CNN Attention': {
        'path': os.path.join(MODELS_DIR, 'cnn_attention_model.keras'),
        'type': 'cnn_attention'
    },
    'LSTM': {
        'path': os.path.join(MODELS_DIR, 'lstm_melspec.keras'),
        'type': 'lstm_mel',
        'norm_path': os.path.join(MODELS_DIR, 'lstm_norm_params.npz')
    },
    'CNN Enhanced': {
        'path': os.path.join(MODELS_DIR, 'cnn_enhanced.keras'),
        'type': 'cnn_enhanced',
        'norm_path': os.path.join(MODELS_DIR, 'norm_params.npz')
    }
}

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =============================================================================
# Custom Layers
# =============================================================================
class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model=64, num_heads=4, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.dropout_rate = dropout_rate
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout_rate)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        q = self.split_heads(self.wq(inputs), batch_size)
        k = self.split_heads(self.wk(inputs), batch_size)
        v = self.split_heads(self.wv(inputs), batch_size)
        scale = tf.math.sqrt(tf.cast(self.depth, tf.float32))
        attention = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / scale, axis=-1)
        attention = self.dropout(attention, training=training)
        out = tf.matmul(attention, v)
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, (batch_size, -1, self.d_model))
        return self.dense(out)
    
    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model, 'num_heads': self.num_heads, 'dropout_rate': self.dropout_rate})
        return config

# =============================================================================
# Feature Extraction - Matches training notebooks
# =============================================================================
def extract_cnn_attention_features(audio, sr):
    """
    For CNN Attention model - segmented mel spectrograms.
    Matches method3_cnn_attention_final.ipynb preprocessing.
    """
    # n_mels=64 for CNN Attention (per training notebook)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=64, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    total_frames = mel_db.shape[1]
    segment_len = total_frames // NUM_SEGMENTS
    
    segments = []
    for i in range(NUM_SEGMENTS):
        start = i * segment_len
        end = start + segment_len
        seg = mel_db[:, start:end]
        # Ensure consistent segment size
        if seg.shape[1] < segment_len:
            seg = np.pad(seg, ((0, 0), (0, segment_len - seg.shape[1])))
        segments.append(seg)
    
    features = np.array(segments)  # (10, 64, segment_len)
    return features

def extract_lstm_mel_features(audio, sr):
    """
    For LSTM model - full mel spectrogram.
    Matches method1_lstm_melspec.ipynb preprocessing.
    """
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db.T  # (time, freq)
    
    # Pad/truncate to target length
    if mel_db.shape[0] < TARGET_LENGTH:
        mel_db = np.pad(mel_db, ((0, TARGET_LENGTH - mel_db.shape[0]), (0, 0)))
    else:
        mel_db = mel_db[:TARGET_LENGTH, :]
    
    return mel_db  # (1291, 128)

def extract_cnn_enhanced_features(audio, sr):
    """
    For CNN Enhanced model - mel + delta + delta2.
    Matches method2_cnn_enhanced.ipynb preprocessing.
    """
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db.T  # (time, freq)
    
    # Pad/truncate
    if mel_db.shape[0] < TARGET_LENGTH:
        mel_db = np.pad(mel_db, ((0, TARGET_LENGTH - mel_db.shape[0]), (0, 0)))
    else:
        mel_db = mel_db[:TARGET_LENGTH, :]
    
    # Compute deltas
    delta = librosa.feature.delta(mel_db.T).T
    delta2 = librosa.feature.delta(mel_db.T, order=2).T
    
    # Stack as 3 channels
    features = np.stack([mel_db, delta, delta2], axis=-1)
    return features  # (1291, 128, 3)

# =============================================================================
# Flask App
# =============================================================================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

models = {}
norm_params = {}

def load_models():
    global models, norm_params
    print("Loading models...")
    
    for name, config in MODEL_CONFIGS.items():
        if os.path.exists(config['path']):
            try:
                if 'attention' in config['type']:
                    model = keras.models.load_model(
                        config['path'], 
                        custom_objects={'MultiHeadAttention': MultiHeadAttention}
                    )
                else:
                    model = keras.models.load_model(config['path'])
                
                models[name] = {'model': model, 'type': config['type']}
                
                # Load normalization params if available
                if 'norm_path' in config and os.path.exists(config['norm_path']):
                    params = np.load(config['norm_path'])
                    norm_params[name] = {'mean': params['mean'], 'std': params['std']}
                    print(f"  ✅ {name} (with norm params)")
                else:
                    print(f"  ✅ {name}")
                    
            except Exception as e:
                print(f"  ❌ {name}: {e}")
    
    print(f"Loaded {len(models)} models")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if not file.filename or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    try:
        filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load audio
        audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION)
        target_len = SAMPLE_RATE * DURATION
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        
        results = {}
        
        for name, config in models.items():
            model = config['model']
            model_type = config['type']
            
            # Extract features based on model type
            if model_type == 'cnn_attention':
                features = extract_cnn_attention_features(audio, sr)
                # Normalize
                features = (features - features.mean()) / (features.std() + 1e-8)
                features = features[..., np.newaxis][np.newaxis, ...]
                
            elif model_type == 'lstm_mel':
                features = extract_lstm_mel_features(audio, sr)
                # Use saved norm params if available
                if name in norm_params:
                    features = (features - norm_params[name]['mean']) / (norm_params[name]['std'] + 1e-8)
                else:
                    features = (features - features.mean()) / (features.std() + 1e-8)
                features = features[np.newaxis, ...]
                
            elif model_type == 'cnn_enhanced':
                features = extract_cnn_enhanced_features(audio, sr)
                # Use saved norm params if available
                if name in norm_params:
                    mean = np.squeeze(norm_params[name]['mean'])
                    std = np.squeeze(norm_params[name]['std'])
                    features = (features - mean) / (std + 1e-8)
                else:
                    features = (features - features.mean()) / (features.std() + 1e-8)
                features = features[np.newaxis, ...]
            
            pred = model.predict(features, verbose=0)
            results[name] = {
                'genre': GENRES[np.argmax(pred[0])],
                'confidence': float(np.max(pred[0]) * 100),
                'all': {g: float(pred[0][i]*100) for i, g in enumerate(GENRES)}
            }
        
        os.remove(filepath)
        return jsonify({'predictions': results})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_models()
    print("\n🎵 GTZAN Genre Classification Demo")
    print("🌐 Open http://localhost:5001\n")
    app.run(debug=False, host='0.0.0.0', port=5001)
