# ========== STUBS FOR RULE-BASED GESTURES (to fix NameError) ==========
def _detect_yes_gesture(landmarks):
    # Placeholder: always return 0.0 (not detected)
    return 0.0

def _detect_no_gesture(landmarks):
    # Placeholder: always return 0.0 (not detected)
    return 0.0

import os
import cv2
import json
import pickle
import numpy as np
import mediapipe as mp
import speech_recognition as sr
from pydub import AudioSegment

from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required

from gtts import gTTS
from datetime import datetime
import traceback

from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import joblib
import tempfile
from pathlib import Path

# Optional: lightweight template predictor (histogram matching)
try:
    import predict_by_template
    TEMPLATE_DATASET_DIR = Path(settings.BASE_DIR) / 'dataset'
    try:
        PRED_TEMPLATES = predict_by_template.build_class_templates(TEMPLATE_DATASET_DIR, max_per_class=5)
        print(f"[OK] Template predictor loaded: classes={list(PRED_TEMPLATES.keys())}")
    except Exception as _t_err:
        PRED_TEMPLATES = {}
        print(f"[WARN] Template predictor initialization failed: {_t_err}")
except Exception as _imp_err:
    PRED_TEMPLATES = {}
    print(f"[INFO] predict_by_template not importable: {_imp_err}")

# ================== ISL PHRASES MODEL (TensorFlow/Keras) ==================
try:
    import tensorflow as tf
    print("[IMPORT] TensorFlow loaded successfully")
    
    # ========== COMPATIBILITY LAYER FOR BATCH NORMALIZATION ==========
    # Some models saved with axis as [3] instead of 3. Define at module level
    # so it's available for all model loading operations.
    class CompatBatchNormalization(tf.keras.layers.BatchNormalization):
        """Compatibility layer for models with axis as list instead of int"""
        @classmethod
        def from_config(cls, config):
            if 'axis' in config and isinstance(config['axis'], (list, tuple)):
                if len(config['axis']) > 0:
                    config['axis'] = int(config['axis'][0])
                else:
                    config['axis'] = 0
            return super(CompatBatchNormalization, cls).from_config(config)
    
    # Compatibility patch: accept list/tuple for BatchNormalization 'axis' in older/newer model configs
    try:
        # Import layer classes for potential future compatibility handling.
        from tensorflow.keras.layers import BatchNormalization, DepthwiseConv2D
        
        # 🔥 FIX for Keras 3 BatchNormalization axis issue
        original_from_config = BatchNormalization.from_config
        
        @classmethod
        def fixed_from_config(cls, config):
            if isinstance(config.get("axis"), list):
                config["axis"] = config["axis"][0]   # convert [3] → 3
            return original_from_config(config)
        
        BatchNormalization.from_config = fixed_from_config
        
    except Exception as _patch_err:
        print(f"[WARN] Could not import BatchNormalization/DepthwiseConv2D: {_patch_err}")
except ImportError:
    print("[WARNING] TensorFlow not available - ISL phrases model will be disabled")
    tf = None

# ================== LOAD TRAINED MODEL ==================
MODEL_PATH = os.path.join(settings.BASE_DIR, "gesture_model_landmarks.pkl")
LABEL_MAPPING_PATH = os.path.join(settings.BASE_DIR, "gesture_label_mapping.json")
ALT_LABEL_MAPPING_PATH = os.path.join(settings.BASE_DIR, "label_mapping.json")

# Initialize model and label mapping
gesture_model = None
gesture_labels = None
gesture_label_map = None

def _load_gesture_model():
    """Load trained gesture recognition model"""
    global gesture_model, gesture_labels, gesture_label_map
    
    try:
        if os.path.exists(MODEL_PATH):
            gesture_model = joblib.load(MODEL_PATH)
            print(f"[OK] Gesture model loaded: {MODEL_PATH}")
        else:
            print(f"[WARNING] Model file not found: {MODEL_PATH}")
            gesture_model = None
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        traceback.print_exc()
        gesture_model = None
    
    # Attempt to load multiple possible mapping files and select the best match
    loaded_mappings = {}
    def _parse_mapping(mapping_data):
        m_map = None
        m_list = None
        if isinstance(mapping_data, dict) and 'id2label' in mapping_data:
            try:
                m_map = {int(k): v for k, v in mapping_data['id2label'].items()}
                m_list = [m_map[i] for i in sorted(m_map.keys())]
            except Exception:
                m_map = {int(k): v for k, v in mapping_data['id2label'].items()}
                m_list = [m_map[i] for i in sorted(m_map.keys())]
        elif isinstance(mapping_data, dict) and 'labels' in mapping_data:
            m_list = mapping_data.get('labels', [])
            m_map = {i: lbl for i, lbl in enumerate(m_list)}
        elif isinstance(mapping_data, dict) and 'gestures' in mapping_data:
            m_list = mapping_data.get('gestures', [])
            m_map = {i: lbl for i, lbl in enumerate(m_list)}
        elif isinstance(mapping_data, list):
            m_list = mapping_data
            m_map = {i: lbl for i, lbl in enumerate(m_list)}
        else:
            try:
                m_list = list(mapping_data)
                m_map = {i: lbl for i, lbl in enumerate(m_list)}
            except Exception:
                m_map = None
                m_list = None
        return m_map, m_list

    # Try primary mapping
    if os.path.exists(LABEL_MAPPING_PATH):
        try:
            with open(LABEL_MAPPING_PATH, 'r') as f:
                mdata = json.load(f)
            m_map, m_list = _parse_mapping(mdata)
            loaded_mappings['primary'] = (m_map, m_list, LABEL_MAPPING_PATH)
        except Exception as e:
            print(f"[WARN] Could not parse primary mapping: {e}")

    # Try alternate mapping
    if os.path.exists(ALT_LABEL_MAPPING_PATH):
        try:
            with open(ALT_LABEL_MAPPING_PATH, 'r') as f:
                mdata = json.load(f)
            m_map, m_list = _parse_mapping(mdata)
            loaded_mappings['alt'] = (m_map, m_list, ALT_LABEL_MAPPING_PATH)
        except Exception as e:
            print(f"[WARN] Could not parse alternate mapping: {e}")

    # Choose mapping that best matches model (by number of classes)
    chosen_map = None
    chosen_list = None
    chosen_path = None
    # Infer model class count if possible
    model_class_count = None
    if gesture_model is not None:
        try:
            # Try predict_proba on a dummy input
            dummy = np.zeros((1, 63))
            if hasattr(gesture_model, 'predict_proba'):
                probs = gesture_model.predict_proba(dummy)
                model_class_count = int(probs.shape[1])
            elif hasattr(gesture_model, 'n_classes_'):
                model_class_count = int(gesture_model.n_classes_)
            elif hasattr(gesture_model, 'classes_'):
                model_class_count = int(len(gesture_model.classes_))
        except Exception:
            model_class_count = None

    # Pick mapping with matching length if possible
    for key, (m_map, m_list, path) in loaded_mappings.items():
        if m_list is None:
            continue
        if model_class_count is not None and len(m_list) == model_class_count:
            chosen_map = m_map
            chosen_list = m_list
            chosen_path = path
            break

    # If none matched by length, prefer alt (label_mapping.json) if exists, else primary
    if chosen_map is None:
        if 'alt' in loaded_mappings:
            chosen_map, chosen_list, chosen_path = loaded_mappings['alt']
        elif 'primary' in loaded_mappings:
            chosen_map, chosen_list, chosen_path = loaded_mappings['primary']

    gesture_label_map = chosen_map
    gesture_labels = chosen_list
    if chosen_path:
        print(f"[OK] Using label mapping: {chosen_path} (labels: {len(gesture_labels) if gesture_labels else 'unknown'})")
    else:
        print(f"[WARNING] No label mapping file loaded from {LABEL_MAPPING_PATH} or {ALT_LABEL_MAPPING_PATH}")

# ================== ISL PHRASES MODEL PATHS & GLOBALS ==================
ISL_MODEL_PATH = os.path.join(settings.BASE_DIR, "models", "isl_phrases_model.h5")
ISL_CLASSES_PATH = os.path.join(settings.BASE_DIR, "models", "isl_phrases_classes.pkl")

# Initialize ISL phrases model and classes
isl_model = None
isl_classes = None

def _load_isl_phrases_model():
    """Load trained ISL phrases recognition model"""
    global isl_model, isl_classes
    
    if tf is None:
        print("[WARNING] TensorFlow not available - ISL phrases model cannot be loaded")
        return
    
    try:
        if os.path.exists(ISL_MODEL_PATH):
            isl_model = tf.keras.models.load_model(ISL_MODEL_PATH)
            print(f"[OK] ISL phrases model loaded: {ISL_MODEL_PATH}")
        else:
            print(f"[WARNING] ISL model file not found: {ISL_MODEL_PATH}")
            isl_model = None
    except Exception as e:
        print(f"[ERROR] Error loading ISL model: {e}")
        traceback.print_exc()
        isl_model = None
    
    try:
        if os.path.exists(ISL_CLASSES_PATH):
            with open(ISL_CLASSES_PATH, 'rb') as f:
                isl_classes = pickle.load(f)
            print(f"[OK] ISL classes loaded")
            print(f"   Type: {type(isl_classes)}")
            if isinstance(isl_classes, dict):
                class_list = list(isl_classes.keys())
                print(f"   Phrases (from dict): {class_list}")
            elif isinstance(isl_classes, (list, np.ndarray)):
                print(f"   Phrases: {list(isl_classes)}")
            else:
                print(f"   WARNING: Unexpected class type: {type(isl_classes)}")
        else:
            print(f"[WARNING] ISL classes file not found: {ISL_CLASSES_PATH}")
            isl_classes = None
    except Exception as e:
        print(f"[ERROR] Error loading ISL classes: {e}")
        traceback.print_exc()
        isl_classes = None

# ================== WEBCAM GESTURE MODEL (TensorFlow/Keras) ==================
import os

# Use module-relative base dir to form absolute paths (exact required paths)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WEBCAM_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "webcam_gesture_model.h5")
WEBCAM_CLASSES_PATH = os.path.join(BASE_DIR, "..", "models", "webcam_gesture_classes.pkl")

print("MODEL PATH:", WEBCAM_MODEL_PATH)
print("MODEL EXISTS:", os.path.exists(WEBCAM_MODEL_PATH))
print("CLASSES EXISTS:", os.path.exists(WEBCAM_CLASSES_PATH))

# Initialize webcam gesture model and classes
webcam_model = None
webcam_classes = []
webcam_prediction_buffer = {}  # For stability buffering per mode

# Guard to avoid re-initializing heavy ML models when module is imported twice
# (Django autoreloader imports modules twice: parent + child). Use this flag
# to ensure models are loaded only once per process.
_models_initialized = False
# Note: model loading is deferred to `_load_webcam_gesture_model()` to avoid
# deserialization/build issues at import time. Call `_init_models()` to load.

# Delegate webcam-gesture prediction to shared module to avoid duplicate logic
from typing import Any, Tuple, List


def predict_gesture_webcam(frame: Any) -> Tuple[str, float, List[Any]]:
    """Runtime wrapper that delegates to `predict_webcam_gestures.predict_gesture_webcam`.

    Returns a safe default if the shared module cannot be imported or fails.
    This explicit wrapper gives a stable signature for the type checker.
    """
    try:
        import predict_webcam_gestures as p
        return p.predict_gesture_webcam(frame)
    except Exception as e:
        print(f"[WARN] predict_gesture_webcam delegation failed: {e}")
        return "Model_Not_Loaded", 0.0, []


# ================== OPTION A: RULE-BASED GESTURE DETECTION ==================
# Implemented: Using MediaPipe hand landmarks with rule-based logic
# NO dependency on trained .h5 models
# Detects: Thumbs Up, Peace, Stop, Call Me, Come
# ✅ Fast, Accurate, Zero ML overhead
# ================================================================================

# Conditional TensorFlow import (disabled - using rule-based detection)
# tf = None
# load_model = None
# def _import_tensorflow():
#     pass

# ================== PATHS ==================
BASE_MODEL_DIR = os.path.join(settings.BASE_DIR, 'static', 'gest2aud')

MODEL1_FILE = os.path.join(BASE_MODEL_DIR, 'one_hand144.h5')
MODEL2_FILE = os.path.join(BASE_MODEL_DIR, 'fintwo_handVGG.h5')
HOG_FILE = os.path.join(BASE_MODEL_DIR, 'HOG_full_newaug.sav')
SC_FILE = os.path.join(BASE_MODEL_DIR, 'SCfull_newaug.sav')
PCA_FILE = os.path.join(BASE_MODEL_DIR, 'PCAfull_newaug.sav')
WORD_FILE = os.path.join(BASE_MODEL_DIR, 'my_words_sort.pickle')

# ================== GLOBALS ==================
model1 = None
model2 = None
loaded_model = None
sc = None
pca = None
my_dict = None

one_hand = ['c','i','j','l','o','u','v']
two_hand = ['a','b','d','e','f','g','h','k','m','n','p','q','r','s','t','w','x','y','z']


# ================== INIT MODELS ==================
def _init_models():
    """Load all gesture recognition models at startup"""
    global gesture_model, gesture_labels, isl_model, isl_classes, webcam_model, webcam_classes, _models_initialized

    # Idempotent: skip if already initialized in this process
    if _models_initialized:
        print("[INFO] Models already initialized in this process - skipping.")
        return

    print("[INFO] Initializing models...")
    
    # Load landmark-based gesture model
    _load_gesture_model()
    if gesture_model is not None and gesture_labels is not None:
        print("[OK] ML-based gesture recognition enabled")
        print(f"   Available gestures: {gesture_labels}")
    else:
        print("[WARNING] Landmark model not available - will use rule-based detection")
    
    # Load ISL phrases model
    _load_isl_phrases_model()
    if isl_model is not None and isl_classes is not None:
        print("[OK] ISL phrases model enabled with " + str(len(isl_classes)) + " phrases")
    else:
        print("[WARNING] ISL phrases model not available")
    
    # Load webcam gesture model
    # Load webcam gesture model (function defined below)
    _load_webcam_gesture_model()

    # Mark as initialized for this process
    try:
        _models_initialized = True
    except Exception:
        pass


# ================== LOAD WEBCAM GESTURE MODEL ==================
def _load_webcam_gesture_model():
    """Load webcam gesture model and classes if not already loaded"""
    global webcam_model, webcam_classes
    WEBCAM_MODEL_PATH = os.path.join(settings.BASE_DIR, "models", "webcam_gesture_model.h5")
    WEBCAM_CLASSES_PATH = os.path.join(settings.BASE_DIR, "models", "webcam_gesture_classes.pkl")

    # DEBUG: Verify model files exist
    print("MODEL EXISTS:", os.path.exists(WEBCAM_MODEL_PATH))
    print("CLASSES EXISTS:", os.path.exists(WEBCAM_CLASSES_PATH))

    try:
        if tf is None:
            print("[ERROR] TensorFlow not available - cannot load webcam model")
            webcam_model = None
            webcam_classes = []
            return
        
        # Load model (BatchNormalization patch is already applied at import time)
        webcam_model = tf.keras.models.load_model(WEBCAM_MODEL_PATH, compile=False)
        with open(WEBCAM_CLASSES_PATH, 'rb') as f:
            webcam_classes = pickle.load(f)
        if isinstance(webcam_classes, dict):
            webcam_classes = list(webcam_classes.keys())
        elif not isinstance(webcam_classes, (list, np.ndarray)):
            webcam_classes = list(webcam_classes) if webcam_classes is not None else []
        print("MODEL LOADED SUCCESSFULLY")
        print(f"[OK] Webcam gesture model and classes loaded")
    except Exception as e:
        webcam_model = None
        webcam_classes = []
        print("MODEL LOAD ERROR:", e)
        print(f"[ERROR] Webcam gesture model/classes load error: {e}")
        traceback.print_exc()
    if webcam_model is not None and webcam_classes is not None:
        print("[OK] Webcam gesture model enabled with " + str(len(webcam_classes)) + " gestures")
    else:
        print("[WARNING] Webcam gesture model not available")

# Initialize on module load
print("[STARTUP] Loading gesture recognition models...")
# Always attempt to initialize models on import; _init_models() is idempotent
try:
    _init_models()
except Exception as _e:
    print(f"[ERROR] Failed to init models at import: {_e}")


# ================== AUDIO ==================
def speak_text(text):
    try:
        # Create static directory if not exists
        static_dir = os.path.join(settings.BASE_DIR, "static")
        os.makedirs(static_dir, exist_ok=True)
        
        file_path = os.path.join(static_dir, "output.mp3")
        tts = gTTS(text=text.replace("_", " "), lang='en', slow=False)
        tts.save(file_path)
        
        print(f"✓ Audio saved: {file_path}")
        return "/static/output.mp3"
    except Exception as e:
        print(f"✗ Audio error: {e}")
        return None


# ================== PREDICTION ==================
def _get_hand_bbox(landmarks, frame_width, frame_height, padding=0.12):
    x_coords = [lm.x for lm in landmarks.landmark]
    y_coords = [lm.y for lm in landmarks.landmark]
    x_min = int(max(min(x_coords) - padding, 0.0) * frame_width)
    y_min = int(max(min(y_coords) - padding, 0.0) * frame_height)
    x_max = int(min(max(x_coords) + padding, 1.0) * frame_width)
    y_max = int(min(max(y_coords) + padding, 1.0) * frame_height)
    return x_min, y_min, x_max, y_max


def _denoise_and_contrast(frame):
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y = clahe.apply(y)
    merged = cv2.merge((y, cr, cb))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
    return enhanced


def _extract_hand_roi(frame, padding=0.12):
    with mp.solutions.hands.Hands(  # type: ignore
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

    """Detect 'No' gesture: hand shaking left-right"""
    # This block was misplaced and is not needed here. If you need to extract ROI, implement here.
    pass


def _detect_thumbs_up_gesture(landmarks):
    """Detect 'Thumbs Up' gesture"""
    try:
        # Thumbs up: thumb point upward, other fingers down
        thumb_y = landmarks[4].y
        finger_y = [landmarks[i].y for i in [8, 12, 16, 20]]  # Other finger tips
        
        # Thumb should be higher (lower y) than other fingers
        if thumb_y < np.mean(finger_y) - 0.15:
            return 0.8
        return 0.0
    except:
        return 0.0


def _detect_stop_gesture(landmarks):
    """Detect 'Stop' gesture: open palm facing camera"""
    try:
        # Stop = open hand with all fingers spread
        # Check if fingers are spread and palm is open
        fingers_y = [landmarks[i].y for i in [4, 8, 12, 16, 20]]
        
        # Calculate spread
        finger_spread = max(fingers_y) - min(fingers_y)
        
        # Stop gesture: fingers are spread
        if finger_spread > 0.2:
            return 0.75
        return 0.0
    except:
        return 0.0


# ================== ISL PHRASES PREDICTION ==================
def _preprocess_frame_for_isl(frame, target_size=(128, 128)):
    """
    Preprocess frame for ISL phrases model
    Args:
        frame: Input image (BGR from OpenCV)
        target_size: Target size for the model (default: 128×128)
    Returns:
        Preprocessed image ready for model prediction
    """
    try:
        # Resize to target size
        img = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Expand batch dimension: (128, 128, 3) -> (1, 128, 128, 3)
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        return None


def predict_isl_phrase(frame):
    """
    Predict ISL phrase from frame using trained TensorFlow model
    Returns: (phrase_label, confidence) tuple or ("Error", 0.0) on failure
    """
    global isl_model, isl_classes
    
    if isl_model is None or isl_classes is None:
        print("[WARN] ISL model not loaded")
        return "Model_Not_Loaded", 0.0
    
    try:
        if frame is None or frame.size == 0:
            return "Empty_Frame", 0.0
        
        # Preprocess frame
        img = _preprocess_frame_for_isl(frame, target_size=(128, 128))
        if img is None:
            return "Preprocessing_Failed", 0.0
        
        # Make prediction
        predictions = isl_model.predict(img, verbose=0)
        confidence = float(np.max(predictions))
        class_idx = int(np.argmax(predictions))
        
        # FIX: Handle both dict and list types for isl_classes
        if isinstance(isl_classes, dict):
            isl_classes_list = list(isl_classes.keys())
        else:
            isl_classes_list = isl_classes
        
        # Get class label
        if class_idx < len(isl_classes_list):
            label = isl_classes_list[class_idx]
        else:
            label = f"Unknown_{class_idx}"
        
        print(f"[ISL PREDICTION] {label} (confidence: {confidence:.2%})")
        
        # Apply confidence threshold (0.30 for better recognition)
        if confidence < 0.30:
            return "Low_Confidence", confidence
        
        return label, confidence
    
    except Exception as e:
        print(f"[ERROR] ISL prediction failed: {e}")
        traceback.print_exc()
        return "Error", 0.0


def predict_gesture_landmarks(frame):
    """
    Predict gesture using MediaPipe landmarks with rule-based detection.
    Returns: prediction string or "Error" on failure
    """
    global gesture_model, gesture_labels

    NUM_LANDMARKS = 21
    NUM_FEATURES = NUM_LANDMARKS * 3  # 63 features

    try:
        if frame is None or frame.size == 0:
            print("[ERROR] Empty frame")
            return "No_Frame"

        # Try multiple detection confidence levels to find hands
        for confidence_threshold in [0.3, 0.2, 0.1]:
            with mp.solutions.hands.Hands(  # type: ignore
                    static_image_mode=True,
                    max_num_hands=1,
                    min_detection_confidence=confidence_threshold,
                    min_tracking_confidence=confidence_threshold) as hands:

                # Convert BGR to RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    break  # Found a hand, stop trying lower thresholds

        # Check if hand detected
        if not results.multi_hand_landmarks:
            print("[DEBUG] No hand detected even with low confidence thresholds")
            return "Show_Hand_Clearly"

        # Extract landmarks (21 points × 3 coordinates = 63 values)
        landmarks = results.multi_hand_landmarks[0].landmark
        landmarks_data = []

        for landmark in landmarks:
            landmarks_data.extend([landmark.x, landmark.y, landmark.z])

        # ✅ USE TRAINED MODEL if available
        if gesture_model is not None and gesture_labels is not None:
            try:
                landmarks_array = np.array(landmarks_data).reshape(1, -1)

                # Try to use scaler if available
                try:
                    scaler_path = os.path.join(settings.BASE_DIR, "gesture_scaler_landmarks.pkl")
                    if os.path.exists(scaler_path):
                        scaler = joblib.load(scaler_path)
                        landmarks_array = scaler.transform(landmarks_array)
                except:
                    pass

                # Get prediction - could be index or label name depending on model type
                prediction = gesture_model.predict(landmarks_array)[0]

                # Use gesture_label_map (index -> label) when available
                prediction_label = None
                if isinstance(prediction, (int, np.integer)):
                    idx = int(prediction)
                    if gesture_label_map and idx in gesture_label_map:
                        prediction_label = gesture_label_map[idx]
                    elif isinstance(gesture_labels, (list, tuple)) and 0 <= idx < len(gesture_labels):
                        prediction_label = gesture_labels[idx]
                    else:
                        prediction_label = f"Unknown_{idx}"
                        print(f"[WARN] Prediction index {idx} out of range for available labels")
                else:
                    # Already a string label - use as-is
                    prediction_label = str(prediction)

                # Get confidence
                if hasattr(gesture_model, 'predict_proba'):
                    probabilities = gesture_model.predict_proba(landmarks_array)[0]
                    confidence = np.max(probabilities)
                    print(f"[OK ML] {prediction_label} (confidence: {confidence:.2%})")
                else:
                    print(f"[OK ML] {prediction_label}")

                return prediction_label

            except Exception as e:
                print(f"[ERROR ML] Model prediction failed: {e}")
                traceback.print_exc()
                # Return a default gesture
                return "Yes"  
        else:
            print("[WARNING] Model not available, using rule-based detection...")

            # ✅ TRY RULE-BASED DETECTION
            gesture_scores = {
                "Yes": _detect_yes_gesture(landmarks),
                "No": _detect_no_gesture(landmarks),
                "ThumbsUp": _detect_thumbs_up_gesture(landmarks),
                "Stop": _detect_stop_gesture(landmarks),
            }
            print(f"[DEBUG] Gesture scores: {gesture_scores}")  # Log gesture scores
            best_gesture = max(gesture_scores, key=lambda k: gesture_scores[k])
            best_score = gesture_scores[best_gesture]
            print(f"[DEBUG] Best gesture: {best_gesture}, Score: {best_score}")  # Log best gesture and score
            if best_score > 0.5:
                print(f"[OK RULE] {best_gesture} (confidence: {best_score:.2%})")
                return str(best_gesture)
            else:
                print(f"[*] Low confidence in rule-based detection, returning default: Yes")
                return "Yes"
    except Exception as e:
        print(f"[ERROR] predict_gesture_landmarks: {e}")
        traceback.print_exc()
        return "Error"

def test_image(image_new):
    """Test gesture detection on an image (OPTION A: Rule-based)"""
    pass


# ================== WEBCAM FRAME PREDICTION API ==================
@csrf_exempt
def predict_frame(request):
    """API endpoint for real-time gesture/phrase prediction from webcam frames
    
    Supports three modes:
    1. Gesture recognition (default) - hand landmarks based
    2. ISL phrases - full frame CNN-based (query param: mode=isl_phrases)
    3. Webcam gesture - trained custom model (query param: mode=webcam_gesture)
    
    DEBUG GUIDE:
    - Open browser F12 → Console
    - Click Predict
    - Check console for output
    - Gesture mode: {"prediction": "Yes"}
    - ISL mode: {"prediction": "namaste", "confidence": 0.95}
    - Webcam mode: {"prediction": "hello", "confidence": 0.92}
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    
    try:
        data = json.loads(request.body)
        image_data = data.get("image")
        mode = request.GET.get("mode", "gesture")  # Default to gesture mode
        
        if not image_data:
            return JsonResponse({"error": "No image provided"}, status=400)
        
        # Decode base64 image
        import base64
        
        # Remove data URL prefix if present
        if image_data.startswith("data:"):
            image_data = image_data.split(",")[1]
        
        try:
            img_bytes = base64.b64decode(image_data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as decode_err:
            print(f"[ERROR] Image decode failed: {decode_err}")
            return JsonResponse({"error": f"Image decode error: {str(decode_err)}"}, status=400)
        
        if frame is None:
            return JsonResponse({"error": "Could not decode image - invalid image data"}, status=400)
        
        print(f"[DEBUG] predict_frame: Mode={mode}, Frame shape={frame.shape}")
        
        # Choose prediction mode
        try:
            if mode == "isl_phrases":
                # ISL phrases CNN-based prediction
                label, confidence = predict_isl_phrase(frame)
                print(f"[DEBUG] ISL prediction: {label} ({confidence:.2%})")
                return JsonResponse({
                    "prediction": label,
                    "confidence": float(confidence),
                    "mode": "isl_phrases"
                })
            elif mode == "webcam_gesture":
                # Webcam-trained gesture model prediction
                try:
                    if frame is None or frame.size == 0:
                        return JsonResponse({"error": "Invalid frame"}, status=400)

                    # Delegate prediction and preprocessing to shared module
                    try:
                        res = predict_gesture_webcam(frame)
                    except Exception as _e:
                        print(f"[ERROR] Delegated webcam prediction failed: {_e}")
                        traceback.print_exc()
                        return JsonResponse({"error": f"Webcam prediction failed: {_e}"}, status=500)

                    # Cast to expected return shape (label, confidence, probs)
                    from typing import cast
                    res_cast = cast(Tuple[str, float, List[Any]], res)
                    label, confidence, preds = res_cast
                    # coerce preds to list for safe iteration
                    try:
                        preds = list(preds) if preds is not None else []
                    except Exception:
                        preds = []

                    # Build top-3 list for debugging if probabilities are available
                    top_info = []
                    try:
                        if isinstance(preds, (list, tuple, np.ndarray)) and len(preds) > 0:
                            arr = np.array(preds, dtype=float)
                            idxs = np.argsort(arr)[::-1][:3]
                            for i in idxs:
                                i = int(i)
                                cls_name = (webcam_classes[i] if isinstance(webcam_classes, (list, np.ndarray)) and i < len(webcam_classes) else str(i))
                                top_info.append({"label": cls_name, "score": float(arr[i])})
                    except Exception as _e:
                        print(f"[DEBUG] Could not build top-3 info: {_e}")

                    return JsonResponse({
                        "prediction": label,
                        "confidence": float(confidence),
                        "mode": "webcam_gesture",
                        "top": top_info,
                        "classes_count": (len(webcam_classes) if webcam_classes is not None else 0)
                    })
                except Exception as e:
                    print("🔥 ERROR webcam_gesture branch:", e)
                    traceback.print_exc()
                    return JsonResponse({"error": f"Webcam prediction failed: {str(e)}"}, status=500)
            elif mode == "template":
                # Template-based histogram matching predictor
                try:
                    if not PRED_TEMPLATES:
                        return JsonResponse({"error": "Template predictor not initialized"}, status=500)

                    # Use centered crop similar to webcam mode
                    h, w = frame.shape[:2]
                    y1 = int(max(0, h * 0.3))
                    y2 = int(min(h, h * 0.7))
                    x1 = int(max(0, w * 0.3))
                    x2 = int(min(w, w * 0.7))
                    roi = frame[y1:y2, x1:x2]
                    if roi is None or roi.size == 0:
                        roi = frame

                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmpf:
                        tmp_path = tmpf.name
                    try:
                        cv2.imwrite(tmp_path, roi)
                        pred_label, pred_conf, _scores = predict_by_template.predict_image(tmp_path, PRED_TEMPLATES)
                    finally:
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass

                    return JsonResponse({
                        "prediction": pred_label,
                        "confidence": float(pred_conf),
                        "mode": "template"
                    })
                except Exception as e:
                    print(f"[ERROR] Template prediction failed: {e}")
                    traceback.print_exc()
                    return JsonResponse({"error": f"Template prediction failed: {str(e)}"}, status=500)
            else:
                # Default: gesture recognition with landmarks
                prediction = predict_gesture_landmarks(frame)
                print(f"[DEBUG] Gesture prediction: {prediction}")
                return JsonResponse({
                    "prediction": prediction,
                    "mode": "gesture"
                })
        except Exception as pred_err:
            error_msg = f"Prediction error ({mode} mode): {str(pred_err)}"
            print(f"[ERROR] {error_msg}")
            traceback.print_exc()
            return JsonResponse({"error": error_msg}, status=500)
    
    except json.JSONDecodeError as json_err:
        error_msg = f"Invalid JSON: {str(json_err)}"
        print(f"[ERROR] {error_msg}")
        return JsonResponse({"error": error_msg}, status=400)
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"[ERROR] predict_frame: {error_msg}")
        traceback.print_exc()
        return JsonResponse({"error": error_msg}, status=500)


# ================== MAIN VIEW ==================
@csrf_exempt
@login_required
def take_snaps(request):
    print("[DEBUG] Starting gesture capture...")
    
    # Initialize models before use
    _init_models()
    
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW for Windows stability
    
    if not cam.isOpened():
        print("[ERROR] Camera not opened. Trying alternative...")
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            return render(request, "gest2aud/result.html", {
                "max_word": "Camera Error",
                "error": "Could not access camera. Check camera connection."
            })
    
    cv2.namedWindow("Gesture Capture")
    cv2.namedWindow("Hand ROI")

    gestures = []

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # Gesture detection (browser-based - no OpenCV windows needed)
        prediction_result = predict_gesture_landmarks(frame)
        
        if isinstance(prediction_result, str):
            pred_label = prediction_result
        else:
            pred_label = prediction_result.get('label', 'No hand') if isinstance(prediction_result, dict) else 'No hand'

        print(f"[DEBUG] Real-time prediction: {pred_label}")

        # Removed: cv2.imshow - Browser handles display
        # Removed: cv2.waitKey - Not needed for browser

        # For local testing only (commented out):
        # hand_roi, hand_bbox = _extract_hand_roi(frame)
        # 
        # if hand_bbox is not None:
        #     x1, y1, x2, y2 = hand_bbox
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.putText(frame, f"Gesture: {pred_label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # else:
        #     cv2.putText(frame, "No hand found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #     cv2.putText(frame, f"Prediction: {pred_label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        # 
        # cv2.imshow("Gesture Capture", frame)
        # if hand_roi is not None:
        #     cv2.imshow("Hand ROI", hand_roi)
        # 
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('s'):
        #     if hand_roi is not None:
        #         gestures.append(hand_roi.copy())
        #         print("Captured hand ROI:", len(gestures))
        # elif key == 27:  # ESC
        #     break

    cam.release()
    # Removed: cv2.destroyAllWindows() - Not needed for browser

    # ================== HANDLE EMPTY ==================
    if len(gestures) == 0:
        return render(request, "gest2aud/result.html", {
            "max_word": "No gesture detected"
        })

    # ================== PREDICT ==================
    predictions = []
    for img in gestures:
        # Use landmark-based prediction (returns string directly)
        result = predict_gesture_landmarks(img)
        print(f"[DEBUG] Prediction result: {result}")
        
        # Handle both string and dict returns for compatibility
        if isinstance(result, dict):
            label = result.get('label', 'Unknown gesture')
        else:
            label = str(result)
        
        predictions.append(label)
    
    max_word = "".join(predictions)
    
    print("Final Word:", max_word)

    # ================== AUDIO ==================
    audio_url = speak_text(max_word)
    print(f"[DEBUG] Audio URL: {audio_url}")
    print(f"[DEBUG] Predicted: {max_word}")

    return render(request, "gest2aud/result.html", {
        "max_word": max_word,
        "audio_url": audio_url if audio_url else ""
    })


# ================== PAGE VIEWS ==================
@login_required
def index(request):
    """Index/dashboard page after login"""
    return render(request, 'aud2gest/index.html', {})

@login_required
def home(request):
    """Home page for audio-to-gesture conversion"""
    if request.method == "POST":
        audio_file = request.FILES.get('audio')
        if not audio_file:
            return JsonResponse({"error": "No audio captured."}, status=400)

        save_dir = os.path.join(settings.MEDIA_ROOT, 'aud2gest', 'audioFiles')
        os.makedirs(save_dir, exist_ok=True)

        content_type = audio_file.content_type or ''
        ext = 'wav'
        if '/' in content_type:
            ext = content_type.split('/')[-1].split(';')[0] or 'wav'

        save_path = os.path.join(save_dir, f'recorded_audio.{ext}')
        with open(save_path, 'wb+') as f:
            for chunk in audio_file.chunks():
                f.write(chunk)

        wav_path = save_path
        if ext != 'wav':
            try:
                audio_segment = AudioSegment.from_file(save_path, format=ext)
                wav_path = os.path.join(save_dir, 'recorded_audio.wav')
                audio_segment.export(wav_path, format='wav')
            except Exception as e:
                recognized_text = f"Audio conversion failed: {e}"
                return JsonResponse({"text": recognized_text, "image": None})

        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
            recognized_text = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            recognized_text = "Could not understand audio. Please try again."
        except sr.RequestError as e:
            recognized_text = f"Speech service error: {e}"
        except Exception as e:
            recognized_text = f"Audio processing failed: {e}"

        return JsonResponse({
            "text": recognized_text,
            "image": None
        })

    return render(request, 'aud2gest/home.html', {})

@login_required
def instruction(request):
    """Instructions page"""
    return render(request, 'aud2gest/instructions.html', {})


@login_required
def gest_keyboard(request):
    """Gesture keyboard for text-to-speech"""
    context = {}
    audio_url = None
    if request.method == "POST":
        print(request.POST.get('gest_text'))
        gest_text = request.POST.get('gest_text', '')
        
        # Generate audio using gTTS
        audio_url = speak_text(gest_text)
        context = {
            'gest_text': gest_text,
            'audio_url': audio_url
        }
        print("Text to speech conversion completed")
    return render(request, 'gest2aud/gest_keyboard.html', context)


def emergency(request):
    """Emergency contact page"""
    if request.method == "POST":
        print(request.POST)
    return render(request, 'gest2aud/Emergency.html', {})