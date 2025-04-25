# Kütüphaneler
import cv2
import random
import numpy as np
from ultralytics import YOLO

# Önceden tanımlanmış değişkenler
confidence_score = 0.1  # Güven skoru
text_color = (255, 255, 255)  # Beyaz yazı rengi
background_color = (0, 0, 0)  # Siyah arka plan
font = cv2.FONT_HERSHEY_SIMPLEX  # Font tipi

# Modeli yükle
model = YOLO("C:\\Users\\Muhammed Emre\\Desktop\\python\\project\\medium_model.pt")  # MODELLERİN YOLUNU DÜZENLE!
labels = model.names  # Etiketler (sınıf isimleri)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in labels]  # Her sınıf için rastgele renkler

# Görseli yükle
image_path = "inference/cevher.png"  # GÖRSELİN YOLUNU DÜZENLE!
image = cv2.imread(image_path)

# Görselin başarıyla yüklenip yüklenmediğini kontrol et
if image is None:
    print("[HATA].. Görsel yüklenemedi!")
else:
    # Görseli yeniden boyutlandır
    target_width = 1080  # Yeni genişlik
    target_height = 700  # Yeni yükseklik
    image = cv2.resize(image, (target_width, target_height))  # Boyutlandırma

    # YOLO modelini görsel üzerinde çalıştır
    results = model(image, verbose=False)[0]

    # Kutu koordinatları, sınıf_id, skoru al
    boxes = np.array(results.boxes.data.tolist())

    # Kutuları çiz
    for box in boxes:
        x1, y1, x2, y2, score, class_id = box
        x1, y1, x2, y2, class_id = int(x1), int(y1), int(x2), int(y2), int(class_id)

        box_color = colors[class_id]  # Sınıf için renk

        if score > confidence_score:  # Güven skoru eşik değerinden büyükse
            # Sınır kutusunu çiz
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

            # Metni hazırlamak
            score = score * 100  # Skoru yüzde formatına dönüştür
            class_name = results.names[class_id]  # Sınıf ismini al
            text = f"{class_name}: %{score:.2f}"  # Metni oluştur

            # Metin boyutu ve arka plan kutusunu hesapla
            label_size, base_line = cv2.getTextSize(text, font, 0.6, 2)  # Font boyutu ve kalınlık
            text_x, text_y = x1, y1 - 10 if y1 - 10 > 10 else y1 + 10  # Yüksekliği kontrol et

            # Daha iyi okunabilirlik için arka plan kutusunu çiz
            cv2.rectangle(
                image,
                (text_x, text_y - label_size[1] - 5),
                (text_x + label_size[0] + 5, text_y + base_line - 5),
                (0, 0, 0, 200),  # Yarı saydam siyah arka plan
                thickness=cv2.FILLED
            )

            # Metni kutu üzerine yerleştir
            cv2.putText(image, text, (text_x, text_y), font, 0.6, text_color, thickness=2)  # Metni ekle

    # Görseli kutularla birlikte görüntüle
    cv2.imshow("Image Result", image)
    cv2.waitKey(0)  # Kullanıcı bir tuşa basana kadar bekle
    cv2.destroyAllWindows()  # Pencereleri kapat

    # Sonucu kaydet
    save_path = "results/sample_image_result.jpg"  # Kaydetme yolu
    cv2.imwrite(save_path, image)  # Görseli kaydet
    print("[BİLGİ].. Görsel " + save_path + " yolunda kaydedildi")
