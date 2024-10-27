import cv2
import tempfile
from datetime import datetime
from pathlib import Path
from bottle_classifier import BottleClassifier  # Assuming the previous code is in bottle_classifier.py

class CameraClassifier:
    def __init__(self):
        self.cap = None
        self.classifier = BottleClassifier()
        self.temp_dir = Path(tempfile.gettempdir()) / "bottle_classifier"
        self.temp_dir.mkdir(exist_ok=True)

    def start_camera(self):
        """Initialize and start the camera capture"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera. Please check if it's connected properly.")

        print("\nCamera started successfully!")
        print("Press 'c' to capture and classify an image")
        print("Press 'q' to quit")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Display the camera feed
            cv2.imshow('Bottle Classifier Camera', frame)

            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):  # Capture when 'c' is pressed
                self.capture_and_classify(frame)
            elif key == ord('q'):  # Quit when 'q' is pressed
                break

        self.cleanup()

    def capture_and_classify(self, frame):
        """Capture the current frame and run classification"""
        # Generate a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = self.temp_dir / f"capture_{timestamp}.jpg"
        
        # Save the frame
        cv2.imwrite(str(img_path), frame)
        print(f"\nImage captured and saved to: {img_path}")

        # Run classification
        print("Running classification...")
        try:
            result = self.classifier.classify_image(str(img_path))
            
            # Display results
            print("\nClassification Results:")
            print(f"Is it a bottle? {'Yes' if result['is_bottle'] else 'No'}")
            print(f"Is it a PET bottle? {'Yes' if result['is_pet_bottle'] else 'No'}")
            print("\nConfidence Scores:")
            print(f"Bottle confidence: {result['confidence_scores']['bottle']:.2f}")
            print(f"PET bottle confidence: {result['confidence_scores']['pet']:.2f}")
            
            # Display result on the frame
            result_text = f"Bottle: {'Yes' if result['is_bottle'] else 'No'}"
            cv2.putText(frame, result_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if result['is_bottle']:
                pet_text = f"PET: {'Yes' if result['is_pet_bottle'] else 'No'}"
                cv2.putText(frame, pet_text, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show the frame with results
            cv2.imshow('Capture Result', frame)
            cv2.waitKey(2000)  # Display result for 2 seconds
            
        except Exception as e:
            print(f"Error during classification: {str(e)}")

    def cleanup(self):
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    print("Initializing Camera Classifier...")
    try:
        camera_classifier = CameraClassifier()
        camera_classifier.start_camera()
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure your camera is connected and working")
        print("2. Install required package: pip3 install opencv-python")
        print("3. Check camera permissions on your system")

if __name__ == "__main__":
    main()