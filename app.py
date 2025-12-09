import os
import base64
import json
import cv2
import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Tuple, List
import re

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration classes
@dataclass
class FeatureConfig:
    method: str = 'hog'
    hog_orientations: int = 9
    hog_cell: Tuple[int, int] = (8, 8)
    hog_block: Tuple[int, int] = (16, 16)
    hog_stride: Tuple[int, int] = (8, 8)
    resize_to: int = 48
    deskew: bool = True
    normalize: bool = True

@dataclass
class SegmentationConfig:
    target_height: int = 48
    min_char_width: int = 4
    min_char_area: int = 30
    pad: int = 2

# Field order for KTP
FIELD_ORDER = [
    'NIK',
    'Nama',
    'Tempat/Tgl Lahir',
    'Jenis Kelamin',
    'Alamat',
    'RT/RW',
    'Kel/Desa',
    'Kecamatan',
    'Agama',
    'Status Perkawinan',
    'Pekerjaan',
    'Kewarganegaraan',
    'Berlaku Hingga'
]

# Global variables
svm_model = None
model_config = None
model_loaded = False
feat_config = None
idx_to_class = {}

# Model directory
MODEL_DIR = os.path.dirname(__file__)
SVM_MODEL_NAME = 'digit_svm_best_ml'


def load_config():
    """Load model configuration"""
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
    """Load SVM model"""
    global svm_model, model_loaded, feat_config, idx_to_class
    
    try:
        model_path = os.path.join(MODEL_DIR, f'{SVM_MODEL_NAME}.xml')
        
        if not os.path.exists(model_path):
            print(f"✗ Model file not found: {model_path}")
            model_loaded = False
            return False
        
        svm_model = cv2.ml.SVM_load(model_path)
        
        # Load feature config
        if model_config:
            feat_config = FeatureConfig(
                method=model_config.get('method', 'hog'),
                hog_orientations=model_config.get('hog_orientations', 9),
                hog_cell=tuple(model_config.get('hog_cell', [8, 8])),
                hog_block=tuple(model_config.get('hog_block', [16, 16])),
                hog_stride=tuple(model_config.get('hog_stride', [8, 8])),
                resize_to=model_config.get('resize_to', 48),
                deskew=model_config.get('deskew', True),
                normalize=model_config.get('normalize', True)
            )
            idx_to_class = {int(k): v for k, v in model_config.get('idx_to_class', {}).items()}
        
        print(f"✓ SVM model loaded from: {model_path}")
        model_loaded = True
        return True
        
    except Exception as e:
        print(f"✗ Error loading SVM model: {str(e)}")
        model_loaded = False
        return False


def _deskew(img: np.ndarray, size: int) -> np.ndarray:
    """Deskew image using moments"""
    try:
        m = cv2.moments(img)
        if abs(m['mu02']) < 1e-2:
            return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        skew = m['mu11'] / m['mu02']
        M = np.float32([[1, skew, -0.5*skew*img.shape[0]], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    except:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def _normalize(feat: np.ndarray) -> np.ndarray:
    """Normalize feature vector"""
    if feat.size == 0:
        return feat
    norm = np.linalg.norm(feat)
    if norm == 0:
        return feat
    return (feat / norm).astype(np.float32)


def extract_feature(img_bin: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """Extract HOG features from binary image"""
    try:
        if img_bin.dtype != np.uint8:
            img_bin = img_bin.astype(np.uint8)
        
        sz = cfg.resize_to
        if cfg.deskew:
            img_in = _deskew(img_bin, sz)
        else:
            img_in = cv2.resize(img_bin, (sz, sz), interpolation=cv2.INTER_AREA)
        
        if cfg.method == 'hog':
            hog = cv2.HOGDescriptor(
                (sz, sz),
                cfg.hog_block,
                cfg.hog_stride,
                cfg.hog_cell,
                cfg.hog_orientations
            )
            feat = hog.compute(img_in)
            if feat is None:
                return np.zeros((0,), dtype=np.float32)
            feat = feat.flatten().astype(np.float32)
        else:
            feat = (img_in.flatten() / 255.0).astype(np.float32)
        
        if cfg.normalize:
            feat = _normalize(feat)
        
        return feat
    except Exception as e:
        print(f"Feature extraction error: {str(e)}")
        return np.zeros((0,), dtype=np.float32)


def standardize_scale(img_bgr, target_width=1200):
    """Standardize image width"""
    H, W = img_bgr.shape[:2]
    if W == target_width:
        return img_bgr
    scale = target_width / float(W)
    new_h = int(round(H * scale))
    return cv2.resize(img_bgr, (target_width, new_h), interpolation=cv2.INTER_AREA)


def crop_info_region(img_bgr):
    """Crop the textual value block (skip header)"""
    img_scaled = standardize_scale(img_bgr, 1200)
    gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 12)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    H, W = morph.shape
    
    # Skip header - start from 15% of image height
    top = int(H * 0.15)
    bottom = H - 10
    
    proj_cols = morph.sum(axis=0)
    col_thresh = 0.01 * 255 * H
    active_cols = np.where(proj_cols > col_thresh)[0]
    
    if len(active_cols) == 0:
        left = int(0.30 * W)
        right = int(0.90 * W)
    else:
        raw_left = max(active_cols[0] - 4, 0)
        raw_right = min(active_cols[-1] + 4, W)
        raw_width = max(1, raw_right - raw_left)
        shift_px = int(raw_width * 0.22)
        focus_width = max(int(raw_width * 0.68), 40)
        left = min(max(0, raw_left + shift_px), max(0, W - focus_width))
        right = left + focus_width
    
    info_roi = img_scaled[top:bottom, left:right]
    return info_roi, (left, top, right - left, bottom - top)


def segment_lines(roi_bgr):
    """Segment horizontal text lines"""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.medianBlur(th, 3)
    H, W = th.shape
    
    k_w = max(20, W // 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 4))
    band = cv2.dilate(th, kernel, iterations=1)
    
    cnts, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        
        if h < 15 or h > 100:
            continue
        if w < 0.20 * W:
            continue
        
        aspect = w / max(h, 1)
        if aspect < 2.5:
            continue
        
        pad = 4
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        w0 = min(W - x0, w + 2 * pad)
        h0 = min(H - y0, h + 2 * pad)
        boxes.append((x0, y0, w0, h0))
    
    boxes.sort(key=lambda b: b[1])
    
    # Merge overlapping boxes
    merged = []
    for box in boxes:
        if not merged:
            merged.append(box)
            continue
        
        prev = merged[-1]
        x, y, w, h = box
        px, py, pw, ph = prev
        overlap_y = max(0, min(py + ph, y + h) - max(py, y))
        
        if overlap_y > 0.4 * min(h, ph):
            new_x = min(px, x)
            new_y = min(py, y)
            new_x2 = max(px + pw, x + w)
            new_y2 = max(py + ph, y + h)
            merged[-1] = (new_x, new_y, new_x2 - new_x, new_y2 - new_y)
        else:
            merged.append(box)
    
    return merged


def assign_field_names(line_boxes):
    """Map detected line boxes to field names"""
    names = FIELD_ORDER.copy()
    if not line_boxes:
        return {}
    
    mapped = {}
    count = min(len(line_boxes), len(names))
    for i in range(count):
        mapped[names[i]] = line_boxes[i]
    return mapped


def _estimate_value_start(bin_img, min_fraction=0.35):
    """Estimate where value starts in binary image"""
    H, W = bin_img.shape
    if W == 0:
        return 0
    
    start = int(W * min_fraction)
    col_sum = bin_img.sum(axis=0) / 255.0
    gap_threshold = 0.14 * H
    search_start = int(W * 0.22)
    search_end = int(W * 0.75)
    gap_candidates = []
    run = None
    
    for idx in range(search_start, min(search_end, W - 1)):
        if col_sum[idx] <= gap_threshold:
            if run is None:
                run = [idx, idx]
            else:
                run[1] = idx
        elif run is not None:
            gap_candidates.append(tuple(run))
            run = None
    
    if run is not None:
        gap_candidates.append(tuple(run))
    
    if gap_candidates:
        best_gap = min(gap_candidates, key=lambda g: abs(((g[0] + g[1]) // 2) - start))
        start = min(best_gap[1] + 1, W - 1)
    
    start = max(int(W * 0.25), start)
    return max(0, start)


def _trim_binary(mask):
    """Trim binary mask to content"""
    if mask.size == 0:
        return None
    cols = np.where(mask.sum(axis=0) > 0)[0]
    rows = np.where(mask.sum(axis=1) > 0)[0]
    if len(cols) == 0 or len(rows) == 0:
        return None
    trimmed = mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    return trimmed, int(cols[0]), int(rows[0])


def _decompose_component(component_mask, origin_x, origin_y, seg_cfg: SegmentationConfig):
    """Split connected component into characters"""
    stack = [(component_mask, origin_x, origin_y, 0)]
    pieces = []
    max_depth = 12
    max_nodes = 80
    
    while stack:
        mask, ox, oy, depth = stack.pop()
        trimmed = _trim_binary(mask)
        if not trimmed:
            continue
        
        sub_mask, dx, dy = trimmed
        ox += dx
        oy += dy
        h, w = sub_mask.shape
        
        if len(pieces) + len(stack) > max_nodes:
            pieces.append(((ox, oy, w, h), sub_mask))
            continue
        
        max_width = int(max(1.35 * h, seg_cfg.target_height + 8))
        if depth >= max_depth or w <= max_width or w <= seg_cfg.min_char_width * 2:
            pieces.append(((ox, oy, w, h), sub_mask))
            continue
        
        col_proj = sub_mask.sum(axis=0) / 255.0
        gap_threshold = max(2.0, 0.16 * h)
        low_indices = np.where(col_proj <= gap_threshold)[0]
        
        if low_indices.size == 0:
            pieces.append(((ox, oy, w, h), sub_mask))
            continue
        
        groups = np.split(low_indices, np.where(np.diff(low_indices) > 1)[0] + 1)
        best_group = max(groups, key=len)
        
        if best_group.size < seg_cfg.min_char_width:
            pieces.append(((ox, oy, w, h), sub_mask))
            continue
        
        gap_start = int(best_group[0])
        gap_end = int(best_group[-1])
        
        if gap_start <= 0 or gap_end >= w - 1:
            pieces.append(((ox, oy, w, h), sub_mask))
            continue
        
        left = sub_mask[:, :gap_start]
        right = sub_mask[:, gap_end + 1:]
        
        if left.size == 0 or right.size == 0:
            pieces.append(((ox, oy, w, h), sub_mask))
            continue
        
        stack.append((right, ox + gap_end + 1, oy, depth + 1))
        stack.append((left, ox, oy, depth + 1))
    
    return pieces


def _prepare_char_canvas(char_mask, seg_cfg: SegmentationConfig):
    """Prepare character on canvas"""
    char_mask = char_mask.astype(np.uint8)
    if char_mask.max() <= 1:
        char_mask = char_mask * 255
    
    h, w = char_mask.shape
    if h == 0 or w == 0:
        size = seg_cfg.target_height + 2 * seg_cfg.pad
        return np.zeros((size, size), dtype=np.uint8)
    
    scale = seg_cfg.target_height / float(h)
    new_w = max(1, int(round(w * scale)))
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(char_mask, (new_w, seg_cfg.target_height), interpolation=interp)
    
    canvas_w = max(seg_cfg.target_height, new_w) + 2 * seg_cfg.pad
    canvas_h = seg_cfg.target_height + 2 * seg_cfg.pad
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    
    y_start = seg_cfg.pad
    x_start = seg_cfg.pad + (canvas_w - new_w) // 2
    canvas[y_start:y_start + seg_cfg.target_height, x_start:x_start + new_w] = resized
    
    return canvas


def _reconstruct_text(preds, boxes):
    """Reconstruct text from predictions and boxes"""
    if not preds:
        return ''
    if not boxes or len(preds) != len(boxes):
        text = ''.join(preds).replace('?', '')
        return re.sub(r'\s{2,}', ' ', text).strip()
    
    order = np.argsort([b[0] for b in boxes])
    ordered_preds = [preds[i] for i in order]
    ordered_boxes = [boxes[i] for i in order]
    widths = [b[2] for b in ordered_boxes]
    avg_width = np.mean(widths) if widths else 0
    
    result = [ordered_preds[0]]
    for idx in range(1, len(ordered_preds)):
        prev_box = ordered_boxes[idx - 1]
        curr_box = ordered_boxes[idx]
        gap = curr_box[0] - (prev_box[0] + prev_box[2])
        if avg_width and gap > max(4, 0.45 * avg_width):
            result.append(' ')
        result.append(ordered_preds[idx])
    
    text = ''.join(result).replace('?', '')
    return re.sub(r'\s{2,}', ' ', text).strip()


def _clean_field_value(name, text):
    """Clean field value based on field type"""
    if not text:
        return ''
    
    value = text.strip()
    value = re.sub(r'\s{2,}', ' ', value)
    
    if name == 'NIK':
        value = re.sub(r'[^0-9]', '', value)
    elif name == 'RT/RW':
        digits = re.sub(r'[^0-9]', '', value)
        if len(digits) >= 6:
            value = f'{digits[:3]} {digits[3:6]}'
        elif len(digits) >= 4:
            value = f'{digits[:3]} {digits[3:]}'
        else:
            value = digits
    elif name == 'Jenis Kelamin':
        value = value.replace('LAKILAKI', 'LAKI-LAKI')
        value = value.replace('LAKI LAKI', 'LAKI-LAKI')
        value = value.replace('PEREMPUAN', 'PEREMPUAN')
        value = re.sub(r'\s*(?:/|\\)?\s*(?:GOL|GOLDARAH)?\s*(A|B|AB|O|A\+|B\+|O\+|AB\+)$', '', value, flags=re.IGNORECASE)
        value = value.strip(' -')
        value = value.replace('LAKI- LAKI', 'LAKI-LAKI')
    elif name == 'Tempat/Tgl Lahir':
        value = value.replace(' ,', ',').replace('  ', ' ')
        parts = value.split(',')
        place = parts[0].strip() if parts else value.split(' ')[0]
        digits = re.sub(r'[^0-9]', '', value)
        if len(digits) >= 8:
            d, m, y = digits[:2], digits[2:4], digits[4:8]
            formatted_date = f'{d}-{m}-{y}'
            value = f'{place}, {formatted_date}'
    elif name == 'Berlaku Hingga':
        value = value.replace('SEUMURHIDUP', 'SEUMUR HIDUP')
    
    return value.strip()


def extract_field_char_images(roi_bgr, field_box, seg_cfg: SegmentationConfig, field_name: str = None):
    """Extract character images from field ROI"""
    x, y, w, h = field_box
    line_roi = roi_bgr[y:y+h, x:x+w]
    gray = cv2.cvtColor(line_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.medianBlur(th, 3)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((1, 3), np.uint8), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
    
    height_full, width_full = th.shape
    if height_full == 0 or width_full == 0:
        return [], []
    
    value_start = _estimate_value_start(th)
    if field_name in ['NIK', 'Nama']:
        col_sum = th.sum(axis=0) / 255.0
        dense_threshold = 0.35 * height_full
        dense_cols = np.where(col_sum >= dense_threshold)[0]
        if dense_cols.size > 0:
            leftmost = int(dense_cols[0])
            value_start = max(0, leftmost - 4)
        else:
            value_start = max(0, int(0.08 * width_full))
    
    value_crop = th[:, value_start:]
    if value_crop.size == 0:
        return [], []
    
    separated = cv2.erode(value_crop, np.ones((2, 2), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    chars = []
    boxes = []
    height_val, width_val = value_crop.shape
    
    min_char_height = int(0.35 * seg_cfg.target_height)
    min_char_width = seg_cfg.min_char_width
    min_area = seg_cfg.min_char_area
    
    for cnt in cnts:
        x0, y0, wc, hc = cv2.boundingRect(cnt)
        
        if wc < min_char_width or hc < min_char_height:
            continue
        if wc * hc < min_area:
            continue
        
        x_pad = max(x0 - 1, 0)
        y_pad = max(y0 - 1, 0)
        x_end = min(width_val, x_pad + wc + 2)
        y_end = min(height_val, y_pad + hc + 2)
        component = value_crop[y_pad:y_end, x_pad:x_end]
        pieces = _decompose_component(component, x_pad, y_pad, seg_cfg)
        
        for (sx, sy, sw, sh), mask in pieces:
            if sw < min_char_width or sh < min_char_height:
                continue
            char_canvas = _prepare_char_canvas(mask, seg_cfg)
            chars.append(char_canvas)
            boxes.append((value_start + sx, sy, sw, sh))
    
    order = np.argsort([b[0] for b in boxes])
    chars = [chars[i] for i in order]
    boxes = [boxes[i] for i in order]
    
    return chars, boxes


def predict_digit_chars(chars: list, feat_cfg: FeatureConfig):
    """Predict character labels"""
    preds = []
    confidences = []
    
    for ch in chars:
        feat = extract_feature(ch, feat_cfg)
        if feat.size == 0:
            preds.append('?')
            confidences.append(0.0)
            continue
        
        feat = feat.reshape(1, -1).astype(np.float32)
        _, pred = svm_model.predict(feat)
        pred_idx = int(pred[0][0])
        
        # Get confidence
        decision = svm_model.predict(feat, flags=cv2.ml.ROW_SAMPLE)
        confidence = float(abs(decision[1][0][0]))
        
        pred_label = idx_to_class.get(pred_idx, '?')
        preds.append(pred_label)
        confidences.append(confidence)
    
    return preds, confidences


@app.route('/', methods=['GET', 'POST'])
def root():
    """Root endpoint"""
    if request.method == 'POST':
        return predict()
    return jsonify({
        'message': 'OCR KTP API - Complete Pipeline',
        'endpoints': {
            'GET /': 'This endpoint',
            'GET /health': 'Health check',
            'POST /predict': 'Extract KTP fields from image',
            'GET /model-info': 'Get model information'
        }
    }), 200


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
    """Extract KTP fields from image"""
    try:
        if not model_loaded:
            return jsonify({'error': 'Model not loaded'}), 500
        
        if svm_model is None:
            return jsonify({'error': 'SVM model is None'}), 500
        
        # Get image from request
        if 'file' in request.files:
            file = request.files['file']
            image_bytes = file.read()
            img_array = np.frombuffer(image_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        elif 'image' in request.get_json(silent=True) or {}:
            image_data = request.json.get('image')
            if not image_data:
                return jsonify({'error': 'No image provided'}), 400
            image_bytes = base64.b64decode(image_data)
            img_array = np.frombuffer(image_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            return jsonify({'error': 'No image provided in request'}), 400
        
        if img_bgr is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Initialize config
        seg_cfg = SegmentationConfig(target_height=48, min_char_width=4, min_char_area=30, pad=2)
        
        # 1. Crop info region
        info_roi, crop_rect = crop_info_region(img_bgr)
        
        # 2. Segment lines
        line_boxes = segment_lines(info_roi)
        
        # 3. Assign field names
        mapped_fields = assign_field_names(line_boxes)
        
        # 4. Extract each field
        field_results = {}
        field_details = {}
        
        for field_name in FIELD_ORDER:
            if field_name not in mapped_fields:
                field_results[field_name] = ''
                field_details[field_name] = {
                    'value': '',
                    'raw': '',
                    'characters': [],
                    'confidences': [],
                    'character_details': []
                }
                continue
            
            box = mapped_fields[field_name]
            
            # Extract character images
            field_chars, char_boxes = extract_field_char_images(info_roi, box, seg_cfg, field_name=field_name)
            
            if not field_chars:
                field_results[field_name] = ''
                field_details[field_name] = {
                    'value': '',
                    'raw': '',
                    'characters': [],
                    'confidences': [],
                    'character_details': []
                }
                continue
            
            # Predict characters
            preds, confidences = predict_digit_chars(field_chars, feat_config)
            
            # Reconstruct text
            raw_value = _reconstruct_text(preds, char_boxes)
            cleaned_value = _clean_field_value(field_name, raw_value)
            
            field_results[field_name] = cleaned_value
            
            # Store details
            char_details = []
            for i, (char, pred, conf) in enumerate(zip(field_chars, preds, confidences)):
                # Encode character image to PNG bytes then to base64
                success, img_bytes = cv2.imencode('.png', char)
                if success:
                    char_base64 = base64.b64encode(img_bytes).decode('utf-8')
                else:
                    char_base64 = ''
                
                char_details.append({
                    'index': i,
                    'character': pred,
                    'confidence': float(conf),
                    'image': f'data:image/png;base64,{char_base64}' if char_base64 else None
                })
            
            field_details[field_name] = {
                'value': cleaned_value,
                'raw': raw_value,
                'characters': preds,
                'confidences': [float(c) for c in confidences],
                'character_count': len(field_chars),
                'character_details': char_details
            }
        
        return jsonify({
            'success': True,
            'fields': field_results,
            'field_details': field_details,
            'crop_info': {
                'left': crop_rect[0],
                'top': crop_rect[1],
                'width': crop_rect[2],
                'height': crop_rect[3]
            },
            'total_fields': len(field_results)
        }), 200
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'type': type(e).__name__,
            'traceback': traceback.format_exc()
        }), 400


@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if not model_config:
        return jsonify({'error': 'Model config not loaded'}), 500
    
    return jsonify({
        'model_name': SVM_MODEL_NAME,
        'model_loaded': model_loaded,
        'fields': FIELD_ORDER,
        'config': {
            'method': model_config.get('method'),
            'classifier': model_config.get('classifier'),
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
            },
            'classes': sorted(idx_to_class.values())
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
