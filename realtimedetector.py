import os
import cv2
import json
import base64
import time
import numpy as np
from inference_sdk import InferenceHTTPClient

# -----------------------------
# Step 0: Define output folders
# -----------------------------
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
output_folder = os.path.join(desktop_path, "mandm_output")
os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# Step 1: Set up Inference Client
# -----------------------------
client = InferenceHTTPClient(
    api_url="http://localhost:9001", # local inference server
    api_key="REDACTED_REVOKE_BEFORE_USE"
)

# -----------------------------
# Step 2: Open Camera
# -----------------------------
cap = cv2.VideoCapture(0)  # Kamera aç (0: varsayılan kamera)
if not cap.isOpened():
    print("❌ Kamera açılamadı.")
    exit()

print("✅ Kamera açıldı. Çıkmak için 'q' tuşuna basın.")

# -----------------------------
# Step 3: Continuous Frame Capture & Inference
# -----------------------------
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Kamera görüntüsü alınamadı.")
        break

    # Frame göster
    cv2.imshow("M&M Detector - Live Feed", frame)

    # Her 3 saniyede bir görüntü al
    if frame_count % 90 == 0:  # (30 FPS x 3 saniye) = 90 frame
        print("📸 Görüntü alındı. Inference çalıştırılıyor...")
        
        # Görüntüyü base64'e çevir
        _, buffer = cv2.imencode(".jpg", frame)
        encoded_image = base64.b64encode(buffer).decode("utf-8")

        try:
            # ✅ Workflow çalıştır
            result = client.run_workflow(
                workspace_name="mam-nv1e6",
                workflow_id="detect-count-and-visualize-2",
                images={
                    "image": encoded_image
                }
            )

            # Eğer sonuç bir listeyse ilk elemanı al
            if isinstance(result, list) and len(result) > 0:
                result_data = result[0]
            else:
                result_data = result
            
            # ✅ Çıktıları kaydet
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            json_output_path = os.path.join(output_folder, f"result_{timestamp}.json")
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=4)
            print(f"✅ JSON çıktısı kaydedildi: {json_output_path}")

            # ✅ Labellenmiş görüntüyü kaydet
            annotated_image_b64 = result_data.get("output_image", "")
            if annotated_image_b64:
                try:
                    annotated_image_bytes = base64.b64decode(annotated_image_b64)
                    annotated_image_path = os.path.join(output_folder, f"annotated_{timestamp}.jpg")
                    with open(annotated_image_path, "wb") as img_file:
                        img_file.write(annotated_image_bytes)
                    print(f"✅ Annotated görüntü kaydedildi: {annotated_image_path}")
                except Exception as e:
                    print(f"❌ Annotated görüntü kaydedilemedi: {e}")

        except Exception as e:
            print(f"❌ Inference hatası: {e}")

    frame_count += 1

    # Çıkış için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🔴 Çıkış yapılıyor...")
        break

# -----------------------------
# Step 4: Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()
print("✅ Kamera kapatıldı.")