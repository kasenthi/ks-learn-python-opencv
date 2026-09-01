# Face detection script that captures and saves images of detected faces.
# Images are stored in a per-person folder under the `faces/` directory,
# making it easy to later train a face recognition model per individual.

import cv2
import os
from datetime import datetime

# Ask for the person's name so images can be organized into a named folder
name = input("Enter your name: ").strip()

# Create a folder for this person under `faces/` (e.g. faces/Alice/)
# exist_ok=True means no error if the folder already exists
save_dir = os.path.join("faces", name)
os.makedirs(save_dir, exist_ok=True)

# Load the Haar Cascade face detector from this project folder.
# Using the bundled XML is more reliable than relying on cv2.data.haarcascades,
# which can be missing or incomplete in some OpenCV installs.
face_cascade = cv2.CascadeClassifier(
    os.path.join(
        os.path.dirname(__file__),
        "haarcascade_frontalface_default.xml"
    )
)

if face_cascade.empty():
    print("ERROR: Could not load face detector from haarcascade_frontalface_default.xml")
    exit()

# Open the default webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit()

print(f"Camera started. Saving images for: {name}")
print("A photo will be captured whenever a face is detected.")
print("Press 'q' to quit.")

# Prevent saving hundreds of images per second
last_capture_time = None
capture_delay = 2  # seconds between captures

while True:
    # Read a frame from the webcam
    success, frame = camera.read()

    if not success:
        print("Error: Could not read frame.")
        break

    # Convert frame to grayscale — Haar Cascade detection works on grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    # scaleFactor=1.1: image scaled down by 10% at each pyramid level
    # minNeighbors=5: minimum detections needed before accepting a face region
    # minSize=(50,50): ignore any face smaller than 50×50 pixels
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(60, 60)
    )

    # Draw a rectangle around every detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Label above the bounding box
        cv2.putText(
            frame,
            "Face detected",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Capture an image if a face is detected
    if len(faces) > 0:
        current_time = datetime.now()

        # Only capture once every few seconds to avoid near-duplicate images
        if (
            last_capture_time is None
            or (current_time - last_capture_time).total_seconds()
            >= capture_delay
        ):
            # Build filename with timestamp, saved inside the person's folder
            filename = os.path.join(
                save_dir,
                current_time.strftime("face_%Y%m%d_%H%M%S.jpg")
            )

            cv2.imwrite(filename, frame)

            print(f"Face detected! Image saved: {filename}")

            last_capture_time = current_time

    # Display webcam feed with face annotations
    cv2.imshow("OpenCV Face Detection", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()

print("Camera stopped.")
