import cv2
import base64
import numpy as np
import time
import serial
from inference_sdk import InferenceHTTPClient

# Kamera başlatılıyor (Daha hızlı başlatma için optimize edildi)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Daha hızlı FPS için çözünürlük düşürüldü
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

# Roboflow Inference Client
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="hdqSvyPvtqTfMrOkiEyU"
)

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

    # Görüntüyü sıkıştırmadan base64'e dönüştürme (FPS optimizasyonu)
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    encoded_image = base64.b64encode(buffer).decode("utf-8")

    # Roboflow modelini çalıştır
    result = client.run_workflow(
        workspace_name="mam-nv1e6",
        workflow_id="detect-count-and-visualize-3",
        images={"image": encoded_image}
    )

    result_data = result[0] if isinstance(result, list) else result
    predictions = result_data.get("predictions", {}).get("predictions", [])

    detected_classes = []  # Algılanan sınıflar

    # Algılanan nesneleri işle
    for pred in predictions:
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        class_name = pred["class"]
        confidence = pred["confidence"]

        detected_classes.append(class_name)

        # Kutunun koordinatları
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)

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
