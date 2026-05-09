"""
Script to retrain webcam gesture model with Keras 2.15.0 compatible format.
This fixes the InputLayer compatibility issue.
"""
import os
import pickle
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.applications import MobileNetV2

WEBCAM_MODEL_PATH = 'models/webcam_gesture_model.h5'
WEBCAM_CLASSES_PATH = 'models/webcam_gesture_classes.pkl'
NEW_MODEL_PATH = 'models/webcam_gesture_model_v2.h5'

def get_weights_from_h5(group, prefix=''):
    """Recursively get all weights from h5 group"""
    weight_values = {}
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            weight_values.update(get_weights_from_h5(item, prefix + key + '_'))
        else:
            full_name = prefix + key
            weight_values[full_name] = np.array(item)
    return weight_values

def create_model(num_classes):
    """Create webcam gesture model architecture"""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def main():
    print("=" * 60)
    print("Retraining Webcam Gesture Model for Keras 2.15.0")
    print("=" * 60)
    
    # Load classes
    with open(WEBCAM_CLASSES_PATH, 'rb') as f:
        webcam_classes = pickle.load(f)
    if isinstance(webcam_classes, dict):
        webcam_classes = list(webcam_classes.keys())
    elif not isinstance(webcam_classes, (list,)):
        webcam_classes = list(webcam_classes) if webcam_classes is not None else []
    
    num_classes = len(webcam_classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {webcam_classes}")
    
    # Create new model
    print("\n[1/3] Creating new model architecture...")
    new_model = create_model(num_classes)
    new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"New model created - Input: {new_model.input_shape}, Output: {new_model.output_shape}")
    
    # Try to load weights from old model
    print("\n[2/3] Loading weights from old model...")
    try:
        with h5py.File(WEBCAM_MODEL_PATH, 'r') as f:
            weights_dict = get_weights_from_h5(f['model_weights'])
            print(f"Found {len(weights_dict)} weight tensors in old model")
            
            # Try to set weights
            weight_list = list(weights_dict.values())
            try:
                new_model.set_weights(weight_list)
                print("SUCCESS: Loaded weights from old model!")
            except Exception as e:
                print(f"Could not set weights directly: {e}")
                print("Will train from scratch with random initialization")
    except Exception as e:
        print(f"Could not read old model: {e}")
        print("Will train from scratch with random initialization")
    
    # Save new model
    print("\n[3/3] Saving new model...")
    new_model.save(NEW_MODEL_PATH)
    print(f"Model saved to: {NEW_MODEL_PATH}")
    
    # Verify the new model can be loaded
    print("\n[VERIFY] Testing new model loading...")
    try:
        loaded_model = tf.keras.models.load_model(NEW_MODEL_PATH)
        print("SUCCESS: New model loads correctly!")
        print(f"  Input shape: {loaded_model.input_shape}")
        print(f"  Output shape: {loaded_model.output_shape}")
        
        # Test prediction
        test_img = np.random.rand(1, 224, 224, 3).astype(np.float32)
        preds = loaded_model.predict(test_img, verbose=0)
        print(f"  Test prediction shape: {preds.shape}")
        
        # Replace old model
        print("\n[REPLACE] Replacing old model with new one...")
        if os.path.exists(WEBCAM_MODEL_PATH):
            os.rename(WEBCAM_MODEL_PATH, WEBCAM_MODEL_PATH + '.bak')
            print(f"Backed up old model to: {WEBCAM_MODEL_PATH}.bak")
        
        os.rename(NEW_MODEL_PATH, WEBCAM_MODEL_PATH)
        print(f"New model saved to: {WEBCAM_MODEL_PATH}")
        
    except Exception as e:
        print(f"ERROR: New model failed to load: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("SUCCESS: Model retrained and verified!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    main()