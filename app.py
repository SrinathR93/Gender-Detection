from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cvlib as cv
import os
import logging

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load gender detection model
try:
    model = load_model('gender_detection.model')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

classes = ['man', 'woman']

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'endpoints': {
            'detect_gender': {'method': 'POST', 'path': '/api/detect-gender'}
        }
    })

@app.route('/api/detect-gender', methods=['POST'])
def detect_gender():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
        
    if 'image' not in request.files:
        logger.error("No image in request.files")
        return jsonify({'error': 'No image provided'}), 400
        
    try:
        file = request.files['image']
        logger.debug(f"Received file: {file.filename}")
        
        image_data = file.read()
        if not image_data:
            logger.error("Empty image data")
            return jsonify({'error': 'Empty image data'}), 400
            
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("Could not decode image")
            return jsonify({'error': 'Invalid image format'}), 400
            
        logger.debug("Detecting faces...")
        faces, confidence = cv.detect_face(frame)
        logger.debug(f"Found {len(faces)} faces")
        
        results = []
        
        for idx, f in enumerate(faces):
            (startX, startY, endX, endY) = f[0], f[1], f[2], f[3]
            face_crop = frame[startY:endY, startX:endX]
            
            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                logger.debug(f"Face {idx} too small, skipping")
                continue
                
            face_crop = cv2.resize(face_crop, (96,96))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)
            
            logger.debug(f"Predicting gender for face {idx}...")
            conf = model.predict(face_crop)[0]
            idx = np.argmax(conf)
            label = classes[idx]
            
            results.append({
                'gender': label,
                'confidence': float(conf[idx]),
                'bounding_box': [int(startX), int(startY), int(endX), int(endY)]
            })
            logger.debug(f"Face {idx}: {label} ({conf[idx]:.2f}%)")
        
        return jsonify({'results': results})
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)