import os
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import json
import base64
import numpy as np
from inference_sdk import InferenceHTTPClient

# -----------------------------
# Step 0: Define output folders
# -----------------------------
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
annotations_folder = os.path.join(desktop_path, "mandm_annotations")
annotated_images_folder = os.path.join(desktop_path, "mandm_annotated_images")
os.makedirs(annotations_folder, exist_ok=True)
os.makedirs(annotated_images_folder, exist_ok=True)

# -----------------------------
# Step 1: Use Tkinter to choose an image file
# -----------------------------
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Select an image for M&M detection",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)
if not file_path:
    messagebox.showinfo("Info", "No file selected. Exiting.")
    exit()

print(f"Selected file: {file_path}")
image_basename = os.path.basename(file_path)
image_name, image_ext = os.path.splitext(image_basename)

# -----------------------------
# Step 2: Set up Inference Client & Run Workflow
# -----------------------------
# ✅ Inference Client'i ayarla
client = InferenceHTTPClient(
    api_url="http://localhost:9001", # local inference server
    api_key="REDACTED_REVOKE_BEFORE_USE"
)

# ✅ Görüntüyü oku ve gönder
with open(file_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

# ✅ Workflow çalıştır
result = client.run_workflow(
    workspace_name="mam-nv1e6",
    workflow_id="detect-count-and-visualize-2",
    images={
        "image": encoded_image
    }
)

# Eğer sonuç bir listeyse, ilk elemanı al
if isinstance(result, list) and len(result) > 0:
    result_data = result[0]
else:
    result_data = result

# -----------------------------
# Step 3: Save the JSON Annotations
# -----------------------------
annotations_file = os.path.join(annotations_folder, f"{image_name}_annotations.json")
with open(annotations_file, "w", encoding="utf-8") as f:
    json.dump(result_data, f, indent=4)
print(f"Annotations saved to: {annotations_file}")

# -----------------------------
# Step 4: Decode and Save Annotated Image
# -----------------------------
annotated_image_b64 = result_data.get("output_image", "")
if annotated_image_b64:
    try:
        annotated_image_bytes = base64.b64decode(annotated_image_b64)
        annotated_image_path = os.path.join(annotated_images_folder, f"{image_name}_annotated.jpg")
        with open(annotated_image_path, "wb") as img_file:
            img_file.write(annotated_image_bytes)
        print(f"Annotated image saved to: {annotated_image_path}")
    except Exception as e:
        print(f"Error decoding annotated image: {e}")
else:
    print("No annotated image available.")

# -----------------------------
# Step 5: Display Results
# -----------------------------
if annotated_image_b64:
    image_array = np.frombuffer(annotated_image_bytes, np.uint8)
    display_img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
else:
    display_img = cv2.imread(file_path)

cv2.imshow("M&M Detector Result", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
