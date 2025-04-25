import cv2
import time
import numpy as np
from ultralytics import YOLO
import os

# Önceden tanımlanmış değişkenler
confidence_score = 0.4  # Başlangıç doğruluk eşiği
text_color_b = (0, 0, 0)  # siyah
text_color_w = (255, 255, 255)  # beyaz
font = cv2.FONT_HERSHEY_SIMPLEX

# Modeli yükle
model = YOLO("C:\\Users\\Muhammed Emre\\Desktop\\python\\project\\medium_model.pt")
labels = model.names

# Kaydedilecek klasör
save_folder = "detected_frames"
os.makedirs(save_folder, exist_ok=True)

# Kamera akışını başlat
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Kamera akışı durduruldu!")
        break

    # Görüntüyü ön işleme (keskinleştirme filtresi ve kontrast artırma)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_eq = cv2.equalizeHist(frame_gray)
    frame_preprocessed = cv2.cvtColor(frame_eq, cv2.COLOR_GRAY2BGR)

    # Modeli çalıştır
    results = model(frame_preprocessed, verbose=False, conf=confidence_score, iou=0.5)[0]
    boxes = np.array(results.boxes.data.tolist())

    detected = False  # Tespit bayrağı

    for box in boxes:
        x1, y1, x2, y2, score, class_id = box
        x1, y1, x2, y2, class_id = int(x1), int(y1), int(x2), int(y2), int(class_id)

        if score > confidence_score:
            detected = True
            class_name = results.names[class_id]
            score_percentage = score * 100

            # Kutu çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            text = f"{class_name}: %{score_percentage:.2f}"
            text_loc = (x1, y1 - 10)
            cv2.putText(frame, text, text_loc, font, 1, text_color_w, thickness=2)

            # Fotoğraf kaydet
            image_path = os.path.join(save_folder, f"frame_{frame_counter}.jpg")
            cv2.imwrite(image_path, frame)

    frame_counter += 1

    # Görüntüyü ekranda göster
    cv2.imshow("Silah Tespiti", frame)

    # Çıkış için 'q' tuşuna basın
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"[INFO] Tespit edilen kareler '{save_folder}' klasörüne kaydedildi.")
