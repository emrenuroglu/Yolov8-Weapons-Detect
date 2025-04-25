import cv2
import cvzone
from ultralytics import YOLO

# Sınıf isimleri, eşik değerleri ve kutu ölçek faktörü
classNames = ["gun", "knife"]
confidence_threshold = 0.4
box_scale = 1.2

# Fotoğraf dosyasını kaynak olarak kullan
img = cv2.imread("inference/cevher.png")

# Görseli 640x640 boyutuna yeniden boyutlandır
img = cv2.resize(img, (640, 640))

# Modeli yükle
model = YOLO("medium_model.pt")

# Resimdeki nesneleri tespit et
results = model(img)

# Tüm tespit sonuçları üzerinde döngü
for result in results:
    # Her bir nesne tespiti için kutu bilgilerini al
    boxes = result.boxes
    for box in boxes:
        conf = box.conf.item()  # Güven oranı
        if conf > confidence_threshold:
            # Tespit edilen nesne kutusunun koordinatlarını alın
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # float to int
            # Kutu içine Gaussian Blur uygula
            img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (7, 7), 0)

            # Genişlik ve yükseklik üzerinde kutu ölçeklendirme işlemi
            width = int((x2 - x1) * box_scale)
            height = int((y2 - y1) * box_scale)
            x1 = max(0, int(x1 + (x2 - x1) * (1 - box_scale) / 2))
            y1 = max(0, int(y1 + (y2 - y1) * (1 - box_scale) / 2))
            x2 = min(img.shape[1], x1 + width)
            y2 = min(img.shape[0], y1 + height)

            # Sınıf indeksini al ve sınıf adıyla güven skorunu yazdır
            cls = int(box.cls[0])
            label = f'{classNames[cls]}: {conf:.2f}'

            # Kutuyu çiz ve etiketi ekle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(img, label, (x1, max(35, y1 - 10)), scale=1, thickness=1, colorR=(0, 0, 255))

# Sonucu göster
cv2.imshow("Detected Objects", img)
cv2.waitKey(0)
cv2.destroyAllWindows()