from flask import Flask, request, jsonify
from ultralytics import YOLO
import requests, cv2
import numpy as np

app = Flask(__name__)
model = YOLO("yolov8n.pt")  # Or 'yolov8s.pt'

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    image_url = data.get('image_url')

    if not image_url:
        return jsonify({'error': 'Missing image_url'}), 400

    try:
        response = requests.get(image_url)
        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        results = model(img)
        person_count = 0
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if model.names[cls_id] == 'person':
                person_count += 1

        return jsonify({'person_count': person_count})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
