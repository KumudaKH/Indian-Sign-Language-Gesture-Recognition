# 🚀 ISL Phrases Model - Quick Setup & Test

## ⚡ 5-Minute Quick Start

### Step 1: Restart Django Server
```bash
python manage.py runserver
```

### Step 2: Open Browser
```
http://127.0.0.1:8000/aud2gest/home
```

### Step 3: Test Gesture Mode (Default)
```
1. Click "Start Webcam"
2. Allow camera permission
3. Position hand in green box
4. Click "Predict"
5. Should see: "YES", "NO", "STOP", etc.
```

### Step 4: Switch to ISL Phrases Mode
```
1. Click "ISL Phrases" button (blue/inactive button on top right)
2. ROI overlay disappears
3. Click "Predict"
4. Should see phrases like: "NAMASTE", "THANK YOU", etc.
5. Confidence score shown below prediction
```

### Step 5: Try Live Mode
```
1. With ISL Phrases selected
2. Click "Live (ON)" button
3. Real-time predictions every 500ms
4. Position your hand and observe predictions updating
5. Click "Live (OFF)" to stop
```

---

## ✅ What Should Happen

### Gesture Mode
- **Input**: Hand gesture in ROI box
- **Output**: Single letter gesture (Yes, No, Stop, ThumbsUp)
- **Speed**: ~1 second per prediction
- **Confidence**: Not shown (rule-based)

### ISL Phrases Mode
- **Input**: Any ISL phrase gesture (hand positioned toward camera)
- **Output**: Phrase name + confidence score
- **Speed**: ~1 second per prediction
- **Confidence**: 0-100%, rejected if < 60%

### Live Mode
- **Input**: Continuous webcam stream
- **Output**: Updated predictions every 500ms
- **Great for**: Demo and testing different gestures

---

## 🔍 Debug Output

Watch your terminal for logs like:
```
[STARTUP] Loading gesture recognition models...
[INFO] Initializing models...
✅ Gesture model loaded: /path/to/gesture_model_landmarks.pkl
✅ ISL phrases model loaded: /path/to/models/isl_phrases_model.h5
✅ ISL classes loaded: 44 phrases
   First 5 phrases: ['asl_alphabet', 'namaste', 'thank_you', 'hello', 'goodbye']
```

In browser console (F12):
```javascript
// Should see logs like:
"Mode switched to: isl_phrases"
"[DEBUG] Gesture prediction: YES"
"[DEBUG] ISL prediction: NAMASTE (0.95)"
```

---

## 🎯 Expected Behavior

### First Time Run
```
[✅] Models load successfully
[✅] Webcam starts
[✅] Can toggle between modes
[✅] Can make single predictions
[✅] Can enable live predictions
```

### ISL Phrase Prediction
```
Image → Preprocess → Model → Prediction
  ↓        ↓           ↓          ↓
Full  Resize to   MobileNetV2  [0.95, 0.03, 0.01, ...]
frame 128×128    inference     ↓
                              NAMASTE
```

---

## 🛠️ Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| "Model_Not_Loaded" error | Check `models/isl_phrases_model.h5` exists |
| Camera won't start | Allow camera permission in browser settings |
| No prediction response | Check Django console for errors |
| Low confidence predictions | Improve lighting, position gesture clearly |
| "ModuleNotFoundError: tensorflow" | Run: `pip install tensorflow` |
| Button doesn't switch modes | Refresh page, check browser console |

---

## 📊 Performance Notes

| Operation | Time |
|-----------|------|
| Load ISL model | 2-3 seconds (on startup) |
| Single prediction | 50-100ms (CPU) |
| Single prediction | 20-50ms (GPU) |
| Live mode predictions | ~2-4 per second |
| Frame processing | Real-time at ~30fps |

---

## 🎬 Test Scenarios

### Scenario 1: Simple Gesture Test
```
1. Start webcam
2. Default mode = Gesture
3. Make hand gesture
4. Click Predict
5. Should recognize gesture
```

### Scenario 2: ISL Phrase Test
```
1. Start webcam
2. Click "ISL Phrases" button
3. Make an ISL gesture (e.g., prayer hands for namaste)
4. Click Predict
5. Should recognize phrase
```

### Scenario 3: Live Demo
```
1. Click "ISL Phrases" button
2. Click "Live (ON)"
3. Slowly show different gestures
4. Watch predictions update in real-time
5. Click "Live (OFF)"
```

### Scenario 4: Mode Switching
```
1. Make prediction in Gesture mode
2. Switch to ISL Phrases mode
3. Same gesture might have different meaning
4. Test both modes on same gesture
```

---

## 💾 File Checklist

Make sure these files exist:
```
✅ models/isl_phrases_model.h5          (68 MB)
✅ models/isl_phrases_classes.pkl       (Small pickle file)
✅ models/isl_phrases_config.json       (Config file)
✅ aud2gest/views.py                    (Updated with ISL code)
✅ templates/aud2gest/home.html         (Updated with UI)
```

Verify:
```bash
ls -lh models/isl_phrases*
```

---

## 🔐 Security Notes

- ✅ CSRF tokens used for all API calls
- ✅ Models validated on load
- ✅ Error handling for invalid inputs
- ✅ Confidence thresholds prevent false positives
- ✅ No sensitive data logged

---

## 📈 Performance Tips

To make it faster:

1. **Use GPU** (if available)
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

2. **Reduce live prediction frequency**
   - Change `setInterval(predictWebcam, 500)` to `1000` (1 second)

3. **Lower image quality**
   - Change `toDataURL('image/jpeg', 0.8)` to `0.6`

4. **Pre-warm model**
   - Run a dummy prediction on startup

---

## 🎓 Learning & Exploring

After basic testing, try:

1. **Check model info**:
   ```bash
   python check_model.py
   ```

2. **Evaluate accuracy**:
   ```bash
   python evaluate_isl_phrases.py
   ```

3. **Batch predict**:
   ```bash
   python predict_isl_phrases.py --image_dir ./test_images/
   ```

4. **Fine-tune model** (if you have more data):
   ```bash
   python train_isl_phrases.py
   ```

---

## 🆘 Still Not Working?

1. **Check logs**: Look at Django console output
2. **Verify files**: `ls models/isl_phrases*`
3. **Check imports**: Open Python and `import tensorflow`
4. **Browser console**: F12 → Console tab for JavaScript errors
5. **Restart server**: Stop and start Django again
6. **Clear cache**: Hard refresh browser (Ctrl+Shift+R)

---

## ✨ Next Improvements

- [ ] Add image upload for prediction
- [ ] Batch process video files
- [ ] Export prediction history
- [ ] Fine-tune model with user corrections
- [ ] Add confidence filtering UI slider
- [ ] Multi-hand support
- [ ] Gesture sequences/combinations

---

**Now you're ready! 🎉**

Start the server, open the app, and test your ISL phrases recognition!

Any issues? Check the ISL_PHRASES_INTEGRATION.md for detailed documentation.
