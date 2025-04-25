import cv2
import numpy as np
from ultralytics import YOLO
import time

confidence_score = 0.6
font = cv2.FONT_HERSHEY_SIMPLEX

# Telefon kamerasının IP adresi
ip_camera_url = "http://172.20.10.7:4747/video"  # IP Webcam'den alınan URL

# Modeli yükle
model = YOLO("C:\\Users\\Muhammed Emre\\Desktop\\python\\project\\medium_model.pt")
labels = model.names

cap = cv2.VideoCapture(ip_camera_url)

# FPS ölçümü için başlangıç zamanı
frame_counter = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Telefon kamerasından görüntü alınamıyor!")
        break

    # --- Frame Boyutunu Küçült ---
    resized_frame = cv2.resize(frame, (640, 480))  # Daha küçük boyut işlem hızını artırır

    # --- YOLO Modeliyle Algılama ---
    results = model(resized_frame, verbose=False, conf=confidence_score, iou=0.5)[0]
    boxes = np.array(results.boxes.data.tolist())

    for box in boxes:
        x1, y1, x2, y2, score, class_id = box
        x1, y1, x2, y2, class_id = int(x1), int(y1), int(x2), int(y2), int(class_id)

        if score > confidence_score:
            class_name = results.names[class_id]
            score_percentage = score * 100
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            text = f"{class_name}: %{score_percentage:.2f}"
            cv2.rectangle(resized_frame, (x1, y1 - 25), (x2, y1), (0, 0, 0), -1)
            cv2.putText(resized_frame, text, (x1, y1 - 10), font, 0.5, (255, 255, 255), 1)

    # --- FPS Hesaplama ---
    frame_counter += 1
    elapsed_time = time.time() - start_time
    fps = frame_counter / elapsed_time
    cv2.putText(resized_frame, f"FPS: {fps:.2f}", (10, 50), font, 1, (0, 255, 0), 2)

    # --- Görüntüyü Ekranda Göster ---
    cv2.imshow("Telefon Kamerası ile Silah Tespiti", resized_frame)

    # --- Çıkış İçin Tuş Kontrolü ---
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
