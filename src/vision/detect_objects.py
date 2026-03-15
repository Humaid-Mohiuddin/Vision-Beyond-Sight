from config import MODEL_PATH
from ultralytics import YOLO

# Load model once (tiny version for real-time)
model = YOLO(MODEL_PATH)

def detect_objects(frame):
    
    # Runs YOLOv8 on the input frame and returns detections + rendered frame.
    
    results = model.predict(frame, imgsz=640, conf=0.4)
    detections = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, class_id]
    annotated_frame = results[0].plot()
    return detections, annotated_frame
