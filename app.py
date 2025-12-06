import os
import base64
import json
import joblib
from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage import exposure
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Model configuration
SVM_MODEL_NAME = os.getenv('SVM_MODEL_NAME', 'digit_svm_best_ml')

# Model directory (local files only)
MODEL_DIR = os.path.dirname(__file__)

# Global variables for model and config
svm_model = None
model_config = None
model_loaded = False


def load_config():
    """Load model configuration from digit_feature_config.json"""
    global model_config
    try:
        config_path = os.path.join(MODEL_DIR, 'digit_feature_config.json')
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        print(f"✓ Configuration loaded from {config_path}")
        return True
    except Exception as e:
        print(f"✗ Error loading config: {str(e)}")
        return False


def load_svm_model():
    """Load SVM model from local file"""
    global svm_model, model_loaded
    
    try:
        # Load from local directory
        model_path = os.path.join(MODEL_DIR, f'{SVM_MODEL_NAME}.xml')
        
        if not os.path.exists(model_path):
            print(f"✗ Model file not found: {model_path}")
            print(f"  Make sure {SVM_MODEL_NAME}.xml exists in {MODEL_DIR}")
            model_loaded = False
            return False
        
        svm_model = cv2.ml.SVM_load(model_path)
        print(f"✓ SVM model loaded from: {model_path}")
        model_loaded = True
        return True
        
    except Exception as e:
        print(f"✗ Error loading SVM model: {str(e)}")
        model_loaded = False
        return False


def deskew_image(img):
    """Deskew image using moments"""
    try:
        # Calculate moments
        m = cv2.moments(cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1])
        
        if abs(m['mu02']) < 1e-2:
            return img
        
        skew = m['mu11'] / m['mu02']
        m_rows, m_cols = img.shape
        
        # Apply affine transformation
        M = np.float32([[1, skew, -0.5 * m_cols * skew],
                       [0, 1, 0]])
        
        img_deskewed = cv2.warpAffine(img, M, (m_cols, m_rows), 
                                       borderMode=cv2.BORDER_REFLECT_101)
        return img_deskewed
    except:
        return img


def extract_hog_features(image):
    """Extract HOG features from image"""
    if model_config is None:
        raise ValueError("Model config not loaded")
    
    try:
        # Get HOG parameters from config
        orientations = model_config.get('hog_orientations', 9)
        cell_size = tuple(model_config.get('hog_cell', [8, 8]))
        block_size = tuple(model_config.get('hog_block', [16, 16]))
        block_stride = tuple(model_config.get('hog_stride', [8, 8]))
        
        # Extract HOG features
        features, hog_image = hog(
            image,
            orientations=orientations,
            pixels_per_cell=cell_size,
            cells_per_block=block_size,
            block_norm='L2-Hys',
            visualize=True,
            multichannel=False
        )
        
        return features.reshape(1, -1)
    except Exception as e:
        raise ValueError(f"Error extracting HOG features: {str(e)}")


def preprocess_digit_image(image_data):
    """Preprocess image for digit recognition (SVM)"""
    try:
        # Convert to grayscale
        if isinstance(image_data, bytes):
            img_array = np.frombuffer(image_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        
        # Deskew if configured
        if model_config.get('deskew', False):
            img = deskew_image(img)
        
        # Resize to configured size
        resize_to = model_config.get('resize_to', 48)
        img = cv2.resize(img, (resize_to, resize_to))
        
        # Normalize if configured
        if model_config.get('normalize', False):
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply threshold
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        return img
        
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model_loaded,
        'config_loaded': model_config is not None
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """Predict digit from image using SVM"""
    try:
        if not model_loaded:
            return jsonify({'error': 'Model not loaded'}), 500
        
        if svm_model is None:
            return jsonify({'error': 'SVM model is None'}), 500
        
        # Get image from request
        if 'file' in request.files:
            file = request.files['file']
            image_bytes = file.read()
        elif 'image' in request.get_json(silent=True) or {}:
            # Base64 encoded image
            image_data = request.json.get('image')
            if not image_data:
                return jsonify({'error': 'No image provided'}), 400
            image_bytes = base64.b64decode(image_data)
        else:
            return jsonify({'error': 'No image provided in request'}), 400
        
        # Preprocess image
        img_preprocessed = preprocess_digit_image(image_bytes)
        
        # Extract HOG features
        features = extract_hog_features(img_preprocessed)
        
        # Make prediction
        features_float = features.astype(np.float32)
        _, predictions = svm_model.predict(features_float)
        
        # Get predicted class
        predicted_idx = int(predictions[0][0])
        idx_to_class = model_config.get('idx_to_class', {})
        predicted_class = idx_to_class.get(str(predicted_idx), str(predicted_idx))
        
        # Get prediction confidence (distance from hyperplane)
        decision = svm_model.predict(features_float, flags=cv2.ml.ROW_SAMPLE)
        confidence = float(abs(decision[1][0][0]))
        
        return jsonify({
            'success': True,
            'predicted_digit': predicted_class,
            'predicted_index': predicted_idx,
            'confidence': confidence,
            'method': model_config.get('method', 'hog')
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e), 'type': type(e).__name__}), 400


@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """Predict multiple digits from multiple images"""
    try:
        if not model_loaded:
            return jsonify({'error': 'Model not loaded'}), 500
        
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        results = []
        
        for file in files:
            try:
                image_bytes = file.read()
                img_preprocessed = preprocess_digit_image(image_bytes)
                features = extract_hog_features(img_preprocessed)
                
                features_float = features.astype(np.float32)
                _, predictions = svm_model.predict(features_float)
                
                predicted_idx = int(predictions[0][0])
                idx_to_class = model_config.get('idx_to_class', {})
                predicted_class = idx_to_class.get(str(predicted_idx), str(predicted_idx))
                
                decision = svm_model.predict(features_float, flags=cv2.ml.ROW_SAMPLE)
                confidence = float(abs(decision[1][0][0]))
                
                results.append({
                    'filename': file.filename,
                    'predicted_digit': predicted_class,
                    'predicted_index': predicted_idx,
                    'confidence': confidence,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': str(e),
                    'success': False
                })
        
        return jsonify({
            'success': True,
            'total': len(files),
            'results': results
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if not model_config:
        return jsonify({'error': 'Model config not loaded'}), 500
    
    num_classes = len(model_config.get('class_to_idx', {}))
    
    return jsonify({
        'model_name': SVM_MODEL_NAME,
        'model_loaded': model_loaded,
        'config': {
            'method': model_config.get('method'),
            'classifier': model_config.get('classifier'),
            'num_classes': num_classes,
            'class_labels': list(model_config.get('class_to_idx', {}).keys()),
            'hog_parameters': {
                'orientations': model_config.get('hog_orientations'),
                'cell_size': model_config.get('hog_cell'),
                'block_size': model_config.get('hog_block'),
                'stride': model_config.get('hog_stride')
            },
            'preprocessing': {
                'deskew': model_config.get('deskew'),
                'normalize': model_config.get('normalize'),
                'resize_to': model_config.get('resize_to')
            }
        }
    }), 200


@app.before_request
def before_request():
    """Initialize models on first request"""
    global model_loaded
    if not model_loaded and request.endpoint not in ['health', 'model_info']:
        load_config()
        load_svm_model()


if __name__ == '__main__':
    # Load config and model at startup
    load_config()
    load_svm_model()
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_ENV') == 'development'
    )
