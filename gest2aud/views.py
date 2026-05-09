import os
import cv2
import json
import pickle
import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization

# 🔥 FIX for Keras 3 / TF 2.15 compatibility
original_from_config = BatchNormalization.from_config

@classmethod
def fixed_from_config(cls, config):
    if isinstance(config.get("axis"), list):
        config["axis"] = config["axis"][0]   # convert [3] → 3
    return original_from_config(config)

BatchNormalization.from_config = fixed_from_config

import mediapipe as mp

from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.core.mail import send_mail

from tensorflow.keras.models import load_model
from gtts import gTTS
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import joblib
import traceback

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
    global model1, model2, loaded_model, sc, pca, my_dict

    if model1 is not None and model2 is not None and loaded_model is not None and sc is not None and pca is not None:
        return

    print(f"[INFO] Current working dir: {os.getcwd()}")
    print(f"[INFO] Initializing models from: {BASE_MODEL_DIR}")
    print(f"[INFO] Checking model files:\n  {MODEL1_FILE}: {os.path.exists(MODEL1_FILE)}\n  {MODEL2_FILE}: {os.path.exists(MODEL2_FILE)}\n  {HOG_FILE}: {os.path.exists(HOG_FILE)}\n  {SC_FILE}: {os.path.exists(SC_FILE)}\n  {PCA_FILE}: {os.path.exists(PCA_FILE)}\n  {WORD_FILE}: {os.path.exists(WORD_FILE)}")

    # Load Keras models only if their files exist
    try:
        if not os.path.exists(MODEL1_FILE) or not os.path.exists(MODEL2_FILE):
            raise FileNotFoundError("One or both Keras model files are missing")

        model1 = load_model(MODEL1_FILE, compile=False)
        model2 = load_model(MODEL2_FILE, compile=False)
        print(f"[INFO] Loaded one hand model and two hand model")
    except Exception as e:
        print(f"[ERROR] Failed to load Keras models: {e}")
        traceback.print_exc()
        model1 = None
        model2 = None

    try:
        if not os.path.exists(HOG_FILE) or not os.path.exists(SC_FILE) or not os.path.exists(PCA_FILE):
            raise FileNotFoundError("HOG/PCA model files are missing")

        loaded_model = joblib.load(HOG_FILE)
        sc = joblib.load(SC_FILE)
        pca = joblib.load(PCA_FILE)
        print(f"[INFO] Loaded HOG + scaler + PCA models")
    except Exception as e:
        print(f"[WARN] Failed to load HOG/PCA models: {e}")
        traceback.print_exc()
        loaded_model = None
        sc = None
        pca = None

    try:
        if os.path.exists(WORD_FILE):
            with open(WORD_FILE, 'rb') as fp:
                my_dict = pickle.load(fp)
            print(f"[INFO] Loaded word mapping")
        else:
            print(f"[WARN] Word mapping file missing: {WORD_FILE}")
            my_dict = None
    except Exception as e:
        print(f"[WARN] Failed to load word mapping: {e}")
        traceback.print_exc()
        my_dict = None


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


def _extract_hand_roi(frame, padding=0.12):
    """
    Extract hand ROI using tight aggressive cropping.
    ROI: 40%-80% height, 30%-70% width
    Forces model to see mostly hands with minimal background.
    """
    h, w, _ = frame.shape
    
    # Calculate tight aggressive ROI boundaries
    roi_top = int(h * 0.4)
    roi_bottom = int(h * 0.8)
    roi_left = int(w * 0.3)
    roi_right = int(w * 0.7)
    
    # Extract ROI
    roi = frame[roi_top:roi_bottom, roi_left:roi_right]
    
    if roi.size == 0:
        return None, None
    
    # Resize to model input
    roi = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)
    roi = _denoise_and_contrast(roi)
    
    # Draw rectangle on frame for visual feedback
    cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)
    cv2.putText(frame, "Gesture Zone", (roi_left + 5, roi_top - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    cv2.imshow("Hand ROI", roi)
    cv2.imwrite("debug_hand.jpg", roi)
    
    return roi, (roi_left, roi_top, roi_right, roi_bottom)


def _denoise_and_contrast(frame):
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y = clahe.apply(y)
    merged = cv2.merge((y, cr, cb))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
    return enhanced

def _preprocess_keras(frame, size):
    img = cv2.resize(frame, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def test_image(image_new):
    _init_models()

    if model1 is None or model2 is None:
        return {"error": "model_load_failure", "detail": "Keras gesture models failed to load."}

    if image_new is None or image_new.size == 0:
        return {"error": "empty_input", "detail": "No hand ROI available for prediction."}

    hand_img = cv2.resize(image_new, (224, 224), interpolation=cv2.INTER_AREA)
    hand_img = _denoise_and_contrast(hand_img)
    cv2.imwrite("debug_hand.jpg", hand_img)
    print(f"[DEBUG] test_image input shape after crop: {hand_img.shape}")

    img_gray = cv2.cvtColor(cv2.resize(hand_img, (28, 28)), cv2.COLOR_BGR2GRAY)
    img1 = _preprocess_keras(hand_img, (144, 144))
    img2 = _preprocess_keras(hand_img, (64, 64))

    try:
        if loaded_model is not None and sc is not None and pca is not None:
            features, _ = hog(img_gray, orientations=8,
                              pixels_per_cell=(4, 4),
                              cells_per_block=(4, 4),
                              block_norm='L2',
                              visualize=True)
            temp = features.reshape(1, -1)
            temp = sc.transform(temp)
            temp = pca.transform(temp)
            z = loaded_model.predict(temp)

            if z[0] == 1.0:
                pred = model2.predict(img2)[0]
                label = two_hand[np.argmax(pred)]
            else:
                pred = model1.predict(img1)[0]
                label = one_hand[np.argmax(pred)]

            confidence = float(np.max(pred))
            print("[DEBUG] Prediction array:", pred)
            print(f"[DEBUG] confidence: {confidence:.4f}, label: {label}")

            # Tiered confidence logic to avoid wrong predictions
            if confidence > 0.75:
                return {'label': label, 'confidence': confidence, 'source': 'hog+keras'}
            elif confidence > 0.5:
                return {'label': 'Adjust hand', 'confidence': confidence, 'source': 'hog+keras'}
            else:
                return {'label': 'Show gesture clearly', 'confidence': confidence, 'source': 'hog+keras'}
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[WARN] HOG/PCA inference failed, falling back to Keras-only: {e}\n{tb}")

    try:
        pred1 = model1.predict(img1)[0]
        pred2 = model2.predict(img2)[0]
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] Keras prediction failed: {e}\n{tb}")
        return {"error": "inference_failure", "detail": str(e)}

    print("[DEBUG] Keras-only pred1:", pred1)
    print("[DEBUG] Keras-only pred2:", pred2)

    max1 = float(np.max(pred1))
    max2 = float(np.max(pred2))

    if max2 > max1:
        label = two_hand[np.argmax(pred2)]
        confidence = max2
    else:
        label = one_hand[np.argmax(pred1)]
        confidence = max1

    print(f"[DEBUG] selected label: {label}, confidence: {confidence:.4f}")
    
    # Tiered confidence logic to avoid wrong predictions
    if confidence > 0.75:
        return {'label': label, 'confidence': confidence, 'source': 'keras-only'}
    elif confidence > 0.5:
        return {'label': 'Adjust hand', 'confidence': confidence, 'source': 'keras-only'}
    else:
        return {'label': 'Show gesture clearly', 'confidence': confidence, 'source': 'keras-only'}


# ================== DEBUG: TEST ENDPOINT ==================
@csrf_exempt
def test_connection(request):
    """Test endpoint - returns hardcoded response to verify backend is reachable"""
    return JsonResponse({"prediction": "WORKING 👍", "status": "Backend connected!"})


@csrf_exempt
def predict_webcam_frame(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        body = request.body.decode('utf-8')
        data = json.loads(body if body else '{}')
        image_data = data.get('image', '')
        if not image_data:
            print('[ERROR] predict_webcam_frame: missing image field')
            return JsonResponse({"error": "No image provided"}, status=400)

        if image_data.startswith('data:'):
            image_data = image_data.split(',', 1)[1]

        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            print(f"[ERROR] predict_webcam_frame: base64 decode failed: {e}")
            return JsonResponse({"error": "Invalid image data"}, status=400)

        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            print('[ERROR] predict_webcam_frame: cv2.imdecode returned None')
            return JsonResponse({"error": "Could not decode image"}, status=400)

        if frame.shape[0] < 32 or frame.shape[1] < 32:
            print(f"[ERROR] predict_webcam_frame: image too small {frame.shape}")
            return JsonResponse({"error": "Image too small for prediction"}, status=400)

        hand_roi, hand_bbox = _extract_hand_roi(frame)
        if hand_roi is None:
            print('[WARN] predict_webcam_frame: no hand ROI detected')
            return JsonResponse({"prediction": "Show gesture clearly"}, status=200)

        result = test_image(hand_roi)
        if not result:
            print('[ERROR] predict_webcam_frame: model inference returned failure')
            return JsonResponse({"error": "Model prediction failed", "detail": "Empty result from test_image."}, status=500)

        if isinstance(result, dict) and result.get('error'):
            print(f"[ERROR] predict_webcam_frame: {result['detail']}")
            return JsonResponse({"error": "Model prediction failed", "detail": result.get('detail', 'Unknown inference error.')}, status=500)

        if isinstance(result, dict):
            try:
                confidence = float(result.get('confidence', 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            # No longer blocking based on confidence - pass through all results
            prediction = result.get('label', 'Show gesture clearly')
        else:
            prediction = str(result)

        return JsonResponse({"prediction": prediction})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ERROR] predict_webcam_frame: {e}\n{tb}")
        return JsonResponse({"error": "Prediction error", "detail": str(e)}, status=500)


# ================== MAIN VIEW ==================
@csrf_exempt
@login_required
def take_snaps(request):
    print("[DEBUG] Starting gesture capture...")
    
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

        hand_roi, hand_bbox = _extract_hand_roi(frame)
        if hand_bbox is not None:
            x1, y1, x2, y2 = hand_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Hand detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand found - show hand clearly", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Gesture Capture", frame)
        if hand_roi is not None:
            cv2.imshow("Hand ROI", hand_roi)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if hand_roi is not None:
                gestures.append(hand_roi.copy())
                print("Captured hand ROI:", len(gestures))
            else:
                print("No hand ROI available to capture. Please show the hand clearly.")

        elif key == 27:  # ESC
            break

    cam.release()
    cv2.destroyAllWindows()

    # ================== HANDLE EMPTY ==================
    if len(gestures) == 0:
        return render(request, "gest2aud/result.html", {
            "max_word": "No gesture detected"
        })

    # ================== PREDICT ==================
    predictions = []
    for img in gestures:
        result = test_image(img)
        if isinstance(result, dict):
            label = result.get('label', 'Show gesture clearly')
            confidence = result.get('confidence', 0.0)
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < 0.6:
                label = "Show gesture clearly"
            predictions.append(label)
        else:
            predictions.append(str(result))

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
    context = {}
    if request.method == "POST":
        try:
            # Collect selected emergency checkboxes
            selections = [v for k, v in request.POST.items() if k.startswith('emergency')]
            if not selections:
                context['message'] = 'No emergency option selected.'
            else:
                # Build email content
                subject = 'Emergency Alert'
                timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
                body_lines = [f'Emergency report received at {timestamp}.', '', 'Selected options:']
                body_lines += [f'- {s}' for s in selections]
                # Optionally include submitting user if available
                if request.user and request.user.is_authenticated:
                    body_lines.insert(1, f'User: {request.user.get_username()}')

                body = '\n'.join(body_lines)

                from_email = settings.EMAIL_HOST_USER
                recipient_list = ['kefa98090@gmail.com']

                # Send email
                send_mail(subject, body, from_email, recipient_list, fail_silently=False)
                context['message'] = f'Alert sent to {recipient_list[0]}.'
                # expose selections and recipient to the template so UI can show them
                context['selections'] = selections
                context['recipient'] = recipient_list[0]
        except Exception as e:
            context['error'] = f'Failed to send alert: {e}'
            # keep selections visible even on error
            try:
                context['selections'] = selections
            except Exception:
                context['selections'] = []
            print(f"[ERROR] emergency send_mail failed: {e}")

    return render(request, 'gest2aud/Emergency.html', context)


@csrf_exempt
def send_emergency_email(request):
    """Example view demonstrating send_mail usage (not wired to template)."""
    try:
        send_mail(
            'Emergency Alert',
            'I need help!',
            settings.EMAIL_HOST_USER,
            [
                'kefa98090@gmail.com',
            ],
            fail_silently=False,
        )
        return JsonResponse({'status': 'sent'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'detail': str(e)}, status=500)