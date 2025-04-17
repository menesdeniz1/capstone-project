import cv2
import numpy as np
import time
import serial
from ultralytics import YOLO

# Load local YOLO model
model = YOLO("best10/best (2).pt")#en iyi best10/best(1) şuanlık ya da best8/bestte iş yapar.

# Kamera başlatılıyor (Daha hızlı başlatma için optimize edildi)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Daha hızlı FPS için çözünürlük düşürüldü
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

# Bluetooth bağlantısı (Arduino'nun bağlı olduğu COM portunu değiştir)
bluetooth_port = "COM9"
baud_rate = 9600

try:
    bluetooth = serial.Serial(bluetooth_port, baud_rate, timeout=1)
    print("Bluetooth bağlantısı kuruldu!")
    time.sleep(2)  # Bağlantının oturması için bekleme süresi
except serial.SerialException as e:
    print("Bluetooth bağlantısı kurulamadı:", e)
    exit()

window_name = "Canlı Nesne Algılama"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# FPS için zaman hesaplamaları
onceki_mesaj = ""
son_gonderim_zamani = time.time()

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı!")
        break

    # YOLO modeli ile tahmin yap
    results = model(frame)
    
    detected_classes = []  # Algılanan sınıflar

    # Algılanan nesneleri işle
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            
            detected_classes.append(class_name)
            
            # Kutunun koordinatları
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Nesne etrafına kutu çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Algılanan nesneleri string olarak hazırla
    mesaj = ",".join(detected_classes) if detected_classes else "NO OBJECT"
    print("Algılanan Nesneler:", mesaj)

    # Bluetooth üzerinden mesaj gönderme (gereksiz tekrarları önleme)
    simdi = time.time()
    if mesaj != onceki_mesaj or (simdi - son_gonderim_zamani) > 2:  # 2 saniyede bir güncelle
        try:
            bluetooth.write((mesaj + "\r\n").encode())  # Bluetooth üzerinden gönder
            onceki_mesaj = mesaj
            son_gonderim_zamani = simdi
        except Exception as e:
            print("Bluetooth veri gönderimi hatası:", e)

    # Sonucu ekranda göster
    cv2.imshow(window_name, frame)

    # Çıkış için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # FPS hesapla ve yazdır
    elapsed = time.time() - start_time
    fps = 1 / elapsed
    print(f"FPS: {fps:.2f}")

# Temizlik
cap.release()
cv2.destroyAllWindows()
bluetooth.close()