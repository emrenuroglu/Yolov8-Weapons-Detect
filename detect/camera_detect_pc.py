import cv2
import numpy as np
from ultralytics import YOLO
import time

confidence_score = 0.6
font = cv2.FONT_HERSHEY_SIMPLEX

# Modeli yükle
model = YOLO("C:\\Users\\Muhammed Emre\\Desktop\\python\\project\\medium_model.pt")
labels = model.names

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_counter = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Kameradan görüntü alınamıyor!")
        break

    # Frame işleme
    results = model(frame, verbose=False, conf=confidence_score, iou=0.5)[0]
    boxes = np.array(results.boxes.data.tolist())

    for box in boxes:
        x1, y1, x2, y2, score, class_id = box
        x1, y1, x2, y2, class_id = int(x1), int(y1), int(x2), int(y2), int(class_id)

        if score > confidence_score:
            class_name = results.names[class_id]
            score_percentage = score * 100
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            text = f"{class_name}: %{score_percentage:.2f}"
            cv2.rectangle(frame, (x1, y1 - 25), (x2, y1), (0, 0, 0), -1)
            cv2.putText(frame, text, (x1, y1 - 10), font, 0.5, (255, 255, 255), 1)

    # FPS hesaplama
    fps = frame_counter / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), font, 1, (0, 255, 0), 2)

    # Ekranda göster
    cv2.imshow("Silah Tespiti", frame)
    frame_counter += 1

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
