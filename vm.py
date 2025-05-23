import cv2
import firebase_admin
from firebase_admin import credentials, storage
from ultralytics import YOLO
import time

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'esp32-cam-detection-efd44.appspot.com'  # Replace with yours
})

def download_image():
    bucket = storage.bucket()
    blob = bucket.blob("data/photo.jpg")
    img_path = "current_image.jpg"
    blob.download_to_filename(img_path)
    print("✅ Downloaded new image from Firebase")
    return img_path

def detect_people(img_path):
    model = YOLO("yolov8n.pt")  # Auto-downloads model if missing
    
    results = model(img_path)
    img = cv2.imread(img_path)
    person_count = 0
    
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if model.names[cls_id] == "person":
            person_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Person {person_count}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    
    output_path = "detected_output.jpg"
    cv2.imwrite(output_path, img)
    print(f"✅ Detected {person_count} people")
    return output_path, person_count

def upload_results(output_path, count):
    bucket = storage.bucket()
    result_blob = bucket.blob(f"results/detected_{int(time.time())}.jpg")
    result_blob.upload_from_filename(output_path)
    print(f"✅ Uploaded results to Firebase. Persons: {count}")

if __name__ == "__main__":
    try:
        img = download_image()
        result, count = detect_people(img)
        upload_results(result, count)
    except Exception as e:
        print(f"❌ Error: {e}")
