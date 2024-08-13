import cv2
import numpy as np
from flask import Flask, request, jsonify
import base64
import os
import mysql.connector

app = Flask(__name__)

def decode_base64_image(base64_string):
    """Decode a base64 image string to a numpy array."""
    img_data = base64.b64decode(base64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

def match_faces(stored_image_path, captured_image):
    """Compare the stored face image with the newly captured face image."""
    stored_image = cv2.imread(stored_image_path)
    
    if stored_image is None:
        return False
    
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(stored_image, None)
    kp2, des2 = orb.detectAndCompute(captured_image, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    match_ratio = len(matches) / min(len(des1), len(des2))

    return match_ratio > 0.4  # Adjust threshold as needed

@app.route('/match', methods=['POST'])
def match():
    stored_image_path = request.json.get('stored_image_path')
    captured_image_base64 = request.json.get('captured_image')

    if not stored_image_path or not captured_image_base64:
        return jsonify({'match': False, 'error': 'Invalid input'}), 400

    captured_image = decode_base64_image(captured_image_base64)

    if match_faces(stored_image_path, captured_image):
        return jsonify({'match': True})
    else:
        return jsonify({'match': False})

def connect_to_database():
    """Connect to the remote database."""
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASS', ''),
        'database': os.getenv('DB_NAME', '')
    }
    return mysql.connector.connect(**db_config)

if __name__ == '__main__':
    port = os.getenv('PORT', 5000)
    app.run(host='0.0.0.0', port=port)

