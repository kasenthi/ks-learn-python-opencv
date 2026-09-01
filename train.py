# Training script for the LBPH face recognizer.
# Reads images from the `faces/` directory (one subfolder per person),
# detects faces in each image, and trains a model saved as `faces.yml`.
# A label map is also written to `names.txt` so recognize.py can look up names.

import cv2
import os
import numpy as np

# Create an LBPH (Local Binary Pattern Histograms) face recognizer.
# LBPH works by describing each pixel by comparing it to its neighbours,
# producing a compact texture histogram that is robust to lighting changes.
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the Haar Cascade detector used to locate faces within each training image
face_cascade = cv2.CascadeClassifier(
    os.path.join(
        os.path.dirname(__file__),
        "haarcascade_frontalface_default.xml"
    )
)

if face_cascade.empty():
    print("ERROR: Could not load face detector")
    exit()


# Accumulators for training data
faces = []   # cropped grayscale face images
labels = []  # integer label corresponding to each face image
names = {}   # mapping of integer label → person's name

# Each person gets a unique integer label, assigned in folder iteration order
label_id = 0

# Walk the `faces/` directory — each subfolder is one person's training set
for name in os.listdir("faces"):

    folder = "faces/" + name

    # Skip any stray files that aren't directories
    if not os.path.isdir(folder):
        continue

    # Register this folder name as a new label
    names[label_id] = name

    # Process every image file inside this person's folder
    for file in os.listdir(folder):

        image = cv2.imread(folder + "/" + file)

        # Skip unreadable or non-image files
        if image is None:
            continue

        # Convert to grayscale — LBPH and Haar Cascade both require grayscale input
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces within the training image
        detected_faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        # Crop each detected face and pair it with the current person's label
        for (x, y, w, h) in detected_faces:
            faces.append(gray[y:y+h, x:x+w])
            labels.append(label_id)

    # Move on to the next person
    label_id += 1


# Train the LBPH recognizer on all collected face crops and their labels
recognizer.train(faces, __import__("numpy").array(labels))

# Persist the trained model to disk so recognize.py can load it at runtime
recognizer.save("faces.yml")

# Write the label → name mapping so recognize.py can display names
with open("names.txt", "w") as f:
    for label, name in names.items():
        f.write(f"{label},{name}\n")

print("Training complete!")
