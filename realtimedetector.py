import cv2
import base64
import numpy as np
import time
from inference_sdk import InferenceHTTPClient

# Kamera başlatılıyor (1280x720 HD çözünürlük)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Kamera açılamadı")
    exit()

# Roboflow Inference Client
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="REDACTED_REVOKE_BEFORE_USE"
)

window_name = "Canli M&M Algilama - HD"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

target_fps = 30
frame_interval = 1 / target_fps

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü JPEG base64'e dönüştür
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_image = base64.b64encode(buffer).decode("utf-8")

    # Workflow'u çalıştır
    result = client.run_workflow(
        workspace_name="mam-nv1e6",
        workflow_id="detect-count-and-visualize-3",
        images={"image": encoded_image}
    )

    result_data = result[0] if isinstance(result, list) else result

    predictions = result_data.get("predictions", {}).get("predictions", [])

    detected_classes = []  # Algılanan sınıfları tutmak için liste

    # Tahminleri ekrana çiz ve konsola yazdırmak için hazırla
    for pred in predictions:
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        class_name = pred["class"]
        confidence = pred["confidence"]

        detected_classes.append(class_name)  # Sınıf adını ekle

        # Kutunun koordinatları
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)

        # Kutuyu çiz
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Etiketi çiz
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Algılanan sınıfları konsola yazdır
    if detected_classes:
        print("Algılanan Nesneler:", ", ".join(detected_classes))
    else:
        print("Algılanan Nesne Yok")

    # Sonucu göster
    cv2.imshow(window_name, frame)

    # Çıkış için 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # FPS'i sabitle (30 FPS)
    elapsed = time.time() - start_time
    if elapsed < frame_interval:
        time.sleep(frame_interval - elapsed)

cap.release()
cv2.destroyAllWindows()