import cv2
import numpy as np
import time
import serial
from ultralytics import YOLO
import threading
import queue

# Configuration parameters for easy tuning
CONFIG = {
    'resolution': (1280, 1024),       # Lower resolution for faster processing
    'confidence_threshold': 0.6,    # Higher confidence threshold to reduce false positives
    'frame_skip': 1,                # Process every n-th frame (1 = process all frames)
    'display_fps': True,            # Display FPS on screen
    'display_detections': True,     # Display detection boxes
    'max_fps_target': 30,           # Target maximum FPS
    'bluetooth_update_interval': 0.5, # Seconds between Bluetooth updates
}

class ObjectDetector:
    def __init__(self, model_path="best4/best.pt", bluetooth_port="COM9", baud_rate=9600):
        # Initialize YOLO model with optimization parameters
        self.model = YOLO(model_path)
        
        # Set model parameters for faster inference
        self.model.conf = CONFIG['confidence_threshold']  # Confidence threshold
        
        # Try to use GPU if available
        self.device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize camera with optimized settings
        self.setup_camera()
        
        # Initialize Bluetooth
        self.setup_bluetooth(bluetooth_port, baud_rate)
        
        # Setup threading components
        self.frame_queue = queue.Queue(maxsize=2)  # Small queue to avoid memory buildup
        self.result_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        
        # Tracking variables
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0
        self.last_bluetooth_time = time.time()
        self.last_message = ""
        
    def setup_camera(self):
        """Initialize and configure the camera with optimized settings"""
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        # Set lower resolution for faster processing
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['resolution'][0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['resolution'][1])
        
        # Additional camera optimizations
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size to reduce latency
        
        if not self.cap.isOpened():
            print("Kamera açılamadı!")
            exit()
            
    def setup_bluetooth(self, port, baud_rate):
        """Initialize Bluetooth connection"""
        try:
            self.bluetooth = serial.Serial(port, baud_rate, timeout=0.5)  # Reduced timeout
            print("Bluetooth bağlantısı kuruldu!")
            # Reduced sleep time for faster startup
            time.sleep(0.5)
        except serial.SerialException as e:
            print("Bluetooth bağlantısı kurulamadı:", e)
            print("Bluetooth olmadan devam ediliyor...")
            self.bluetooth = None
            
    def capture_frames(self):
        """Thread function to capture frames from camera"""
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                print("Kamera görüntüsü alınamadı!")
                break
                
            # Skip frames if needed
            self.frame_count += 1
            if self.frame_count % CONFIG['frame_skip'] != 0:
                continue
                
            # If queue is full, remove oldest frame to avoid blocking
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                    
            # Put new frame in queue
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                pass
                
    def process_frames(self):
        """Thread function to process frames with YOLO model"""
        while not self.stop_event.is_set():
            try:
                # Get frame from queue with timeout to avoid blocking forever
                frame = self.frame_queue.get(timeout=0.1)
                
                # Run inference with optimized parameters
                results = self.model(frame, device=self.device, verbose=False)
                
                # Put results in queue, skip if full
                if not self.result_queue.full():
                    self.result_queue.put((frame, results), block=False)
                    
            except queue.Empty:
                continue
            except queue.Full:
                continue
            except Exception as e:
                print(f"Error in processing: {e}")
                
    def display_and_communicate(self):
        """Thread function to display results and handle Bluetooth communication"""
        window_name = "Canlı Nesne Algılama"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        while not self.stop_event.is_set():
            try:
                # Get processed frame and results with timeout
                frame, results = self.result_queue.get(timeout=0.1)
                
                detected_classes = []
                
                # Process detection results
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.model.names[class_id]
                        
                        detected_classes.append(class_name)
                        
                        # Only draw boxes if enabled
                        if CONFIG['display_detections']:
                            # Kutunun koordinatları
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            
                            # Nesne etrafına kutu çiz
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{class_name}: {confidence:.2f}"
                            cv2.putText(frame, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Calculate and display FPS
                current_time = time.time()
                elapsed = current_time - self.last_fps_time
                if elapsed > 0.5:  # Update FPS every 0.5 seconds
                    self.fps = 1.0 / (elapsed / self.frame_count)
                    self.last_fps_time = current_time
                    self.frame_count = 0
                
                if CONFIG['display_fps']:
                    cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display the frame
                cv2.imshow(window_name, frame)
                
                # Handle Bluetooth communication at reduced frequency
                self.handle_bluetooth(detected_classes)
                
                # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_event.set()
                    break
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in display: {e}")
                
    def handle_bluetooth(self, detected_classes):
        """Handle Bluetooth communication with throttling"""
        if self.bluetooth is None:
            return
            
        current_time = time.time()
        # Only send updates at specified interval or when classes change
        if (current_time - self.last_bluetooth_time >= CONFIG['bluetooth_update_interval']):
            message = ",".join(detected_classes) if detected_classes else "NO OBJECT"
            
            # Only send if message changed or update interval passed
            if message != self.last_message or (current_time - self.last_bluetooth_time >= 2):
                try:
                    self.bluetooth.write((message + "\r\n").encode())
                    self.last_message = message
                    self.last_bluetooth_time = current_time
                except Exception as e:
                    print("Bluetooth veri gönderimi hatası:", e)
                    
    def run(self):
        """Main method to run the detection system with threads"""
        # Create and start threads
        capture_thread = threading.Thread(target=self.capture_frames)
        process_thread = threading.Thread(target=self.process_frames)
        display_thread = threading.Thread(target=self.display_and_communicate)
        
        capture_thread.daemon = True
        process_thread.daemon = True
        display_thread.daemon = True
        
        capture_thread.start()
        process_thread.start()
        display_thread.start()
        
        try:
            # Keep main thread alive
            while not self.stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Kullanıcı tarafından durduruldu")
        finally:
            # Cleanup
            self.stop_event.set()
            capture_thread.join(timeout=1)
            process_thread.join(timeout=1)
            display_thread.join(timeout=1)
            self.cleanup()
            
    def cleanup(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        if self.bluetooth is not None:
            self.bluetooth.close()
            
# Main execution
if __name__ == "__main__":
    detector = ObjectDetector()
    detector.run()
