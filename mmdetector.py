import cv2
import base64
import requests
import time
import numpy as np

# Kamera başlat
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Kamera açılamadı.")
    exit()

# Roboflow workflow bilgileri
workspace_name = "mam-nv1e6"                  # workspace adın
workflow_id = "detect-count-and-visualize-2" # workflow ID
api_key = "REDACTED_REVOKE_BEFORE_USE"             # API anahtarın
server_url = "http://localhost:9001"         # yerel Roboflow sunucusu

window_name = "Roboflow Canli Kamera"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

fps_limit = 15  # FPS sınırı (ideal performans için)

headers = {"Content-Type": "application/json"}

try:
    while True:
        start_time = time.time()

        # Kameradan görüntü al
        ret, frame = cap.read()
        if not ret:
            print("❌ Görüntü okunamadı.")
            break

        # Görüntüyü base64'e dönüştür
        _, buffer = cv2.imencode(".jpg", frame)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        # Workflow URL'yi doğru şekilde belirle
        url = f"{server_url}/{workspace_name}/{workflow_id}?api_key={api_key}"

        # JSON gövdesini oluştur
        data = {
            "images": {
                "image": img_base64
            }
        }

        # Workflow'u çağır (POST request)
        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 200:
            result = response.json()

            predictions = result.get("predictions", {}).get("predictions", [])

            # Her tahmin için bounding box ve label çiz
            for pred in predictions:
                x, y = pred["x"], pred["y"]
                w, h = pred["width"], pred["height"]
                class_name = pred.get("class", "Nesne")
                confidence_score = pred.get("confidence", 0)

                # Bounding box koordinatları
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)

                # Kutu ve label çizimi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name}: {confidence_score * 100:.1f}%"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print(f"❌ Roboflow hata: {response.status_code}, {response.text}")

        # Sonuç görüntüsünü göster
        cv2.imshow(window_name, frame)

        # 'q' ile çıkış
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # FPS kontrolü
        elapsed = time.time() - start_time
        sleep_time = max(1 / fps_limit - elapsed, 0)
        time.sleep(sleep_time)

except Exception as e:
    print(f"❌ Bir hata oluştu: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
