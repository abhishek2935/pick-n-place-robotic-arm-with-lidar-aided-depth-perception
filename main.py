import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
bbox = cv2.selectROI("Select Object", frame, False)

tracker = cv2.legacy.TrackerKCF_create()  
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    success, bbox = tracker.update(frame)
    if success:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
