from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def test_yolov8(image_path):
    # Load the trained model
    model = YOLO('runs/detect/train/weights/best.pt')  # Path to best model

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Run inference
    results = model.predict(image, conf=0.55, iou=0.4)  # Use tuned conf and iou

    # Plot results using OpenCV
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        classes = result.boxes.cls.cpu().numpy()  # Class IDs
        confs = result.boxes.conf.cpu().numpy()   # Confidence scores

        for box, cls, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            color = (0, 255, 0)  # Green for bounding boxes

            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, color, 2)

    # Convert BGR to RGB for display with matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image using matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(image_rgb)
    plt.axis('off')  # Hide axis
    plt.title("YOLOv8 Detection Result")
    plt.show()

if __name__ == "__main__":
    # Specify the path to the test image
    image_path = "C:\\Users\\Kutsa\\Desktop\\mm.jpg"  # Example: 'data/test/image1.jpg'
    test_yolov8(image_path)
