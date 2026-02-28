from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')  # Auto-downloads nano model
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # Auto-detect AND track (no manual selection!)
    results = model.track(frame, persist=True, tracker="botsort.yaml")
    annotated_frame = results[0].plot()  # Draws boxes + track IDs
    
    cv2.imshow("Auto Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
