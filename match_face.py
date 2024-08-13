from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import pymysql

app = Flask(__name__)

def connect_db():
    return pymysql.connect(host='localhost',
                           user='u475049814_kiitattendence',
                           password='CKTMcb@12',
                           db='u475049814_kiitattendence')

@app.route('/match-face', methods=['POST'])
def match_face():
    data = request.get_json()
    student_id = data['student_id']
    captured_image_data = base64.b64decode(data['captured_image'])

    # Convert base64 string to numpy array
    nparr = np.frombuffer(captured_image_data, np.uint8)
    captured_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Retrieve stored image from database
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT face_image_path FROM registered_students WHERE id = %s", (student_id,))
    row = cursor.fetchone()
    stored_image_path = row[0]
    stored_image = cv2.imread(stored_image_path)

    # Perform face matching using OpenCV
    match_result = perform_face_matching(stored_image, captured_image)

    return jsonify({'match': match_result})

def perform_face_matching(stored_image, captured_image):
    # Implement face matching logic using OpenCV
    gray_stored = cv2.cvtColor(stored_image, cv2.COLOR_BGR2GRAY)
    gray_captured = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)

    # Initialize the face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the stored image
    faces_stored = face_cascade.detectMultiScale(gray_stored, 1.1, 4)
    faces_captured = face_cascade.detectMultiScale(gray_captured, 1.1, 4)

    # Assume single face per image
    if len(faces_stored) == 1 and len(faces_captured) == 1:
        (x1, y1, w1, h1) = faces_stored[0]
        (x2, y2, w2, h2) = faces_captured[0]

        roi_stored = gray_stored[y1:y1+h1, x1:x1+w1]
        roi_captured = gray_captured[y2:y2+h2, x2:x2+w2]

        # Use Histogram of Oriented Gradients (HOG) or another feature comparison method
        if np.array_equal(roi_stored, roi_captured):
            return True

    return False

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
