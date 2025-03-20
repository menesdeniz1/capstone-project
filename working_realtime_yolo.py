import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load the trained YOLOv8 model
    model = YOLO('C:/Users/enes/Desktop/capstone/runs/detect/train4/weights/best.pt')  # Path to your trained model

    # Open the webcam (0 = default webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Couldn't open camera.")
        return

    plt.ion()  # Turn on interactive mode for real-time updating
    fig, ax = plt.subplots()

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Plot the results on the frame
        annotated_frame = results[0].plot()

        # Convert BGR to RGB (OpenCV uses BGR, matplotlib expects RGB)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display using matplotlib
        ax.clear()
        ax.imshow(annotated_frame)
        ax.set_xticks([])  # Hide x-axis ticks
        ax.set_yticks([])  # Hide y-axis ticks
        plt.pause(0.01)  # Small pause to allow real-time display

        # Exit on pressing 'q'
        if plt.waitforbuttonpress(0.01) and plt.get_current_fig_manager().canvas.manager.key_press_handler_id:
            print("Stopping...")
            break

    # Release resources
    cap.release()
    plt.close()

if __name__ == "__main__":
    main()
