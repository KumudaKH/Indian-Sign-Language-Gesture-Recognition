# 🚀 QUICK START: Webcam Gesture Training (5 minutes)

## Step-by-Step

### 1️⃣ Collect Webcam Images

```bash
python collect_camera_data.py --images-per-gesture 50
```

**Do this for each gesture:**
- Press **SPACE** to capture images
- Press **n** to next gesture
- Press **q** to quit
- Target: 50-100 images per gesture

**Tip:** Plain background, good lighting, keep hands centered

---

### 2️⃣ Train Model

```bash
python train_webcam_gestures.py
```

**Output:**
- `models/webcam_gesture_model.h5` (trained model)
- `models/webcam_gesture_classes.pkl` (gesture labels)

Takes ~5-15 minutes

---

### 3️⃣ Test with Webcam

```bash
python predict_webcam_gestures.py
```

**See live predictions:**
- Shows gesture name and confidence
- Press **q** to quit
- Press **s** to toggle stability

---

### 4️⃣ Use in Django App

Model **automatically loads** when Django starts!

In your code:
```javascript
// Make prediction with your trained model
fetch('/api/predict-webcam/?mode=webcam_gesture', {
    method: 'POST',
    body: JSON.stringify({image: base64_frame})
});
// Returns: {"prediction": "call_me", "confidence": 0.92}
```

---

## 🎯 That's It!

Your custom gesture model is now trained and integrated! 🎉

**Three modes available:**
- `mode=gesture` → Hand landmarks (original)
- `mode=isl_phrases` → Full frame phrases
- `mode=webcam_gesture` → **Your custom trained model** ← NEW!

---

## 📖 Full Guide

See: `WEBCAM_TRAINING_GUIDE.md`
