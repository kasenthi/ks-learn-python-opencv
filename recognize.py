# Face recognition camera that announces recognized people via text-to-speech.
# Uses a pre-trained LBPH model (faces.yml) and speaks using the macOS `afplay` command.
# Designed to alert "Mama" when a known or unknown face is detected on camera.

import cv2
import os
import asyncio
import edge_tts
import time
import threading
import miniaudio

def play_file(path):
    stream = miniaudio.stream_file(path)
    with miniaudio.PlaybackDevice() as device:
        device.start(stream)
        while device.running:
            time.sleep(0.05)

# Tracks the last name spoken and when it was spoken, to avoid repeating announcements
last_name = ""
last_spoken = 0

# Load the pre-trained LBPH (Local Binary Pattern Histograms) face recognizer model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("faces.yml")

# Load names
# Map from integer label → person's name, populated from names.txt (format: "0,Alice")
names = {}

with open("names.txt", "r") as f:
    for line in f:
        label, name = line.strip().split(",", 1)
        names[int(label)] = name


# NOTE: This function is defined twice (duplicate). Only the second definition takes effect.
def speak_to_mama(name):
    # Skip if audio is already playing to prevent overlapping sounds
    if audio_playing:
        return

    # Run audio playback in a background daemon thread so the camera loop isn't blocked
    threading.Thread(
        target=play_audio,
        args=(name,),
        daemon=True
    ).start()


def speak_to_mama(name):
    # Skip if audio is already playing to prevent overlapping sounds
    if audio_playing:
        return

    # Run audio playback in a background daemon thread so the camera loop isn't blocked
    threading.Thread(
        target=play_audio,
        args=(name,),
        daemon=True
    ).start()


def play_audio(name):
    global audio_playing

    # Lock audio so only one announcement plays at a time
    audio_playing = True

    try:

        # -------------------------
        # UNKNOWN PERSON
        # -------------------------
        if name == "Unknown":

            speech_file = "audio/unknown.mp3"

            # Generate the spoken announcement text for an unrecognized face
            text = "Mamaaa... I don't know who that is..."

            # Use edge_tts to synthesize speech with a child-like voice (slow rate, higher pitch)
            communicate = edge_tts.Communicate(
                text,
                "en-US-AnaNeural",
                rate="-25%",
                pitch="+20Hz"
            )

            # Save the generated speech to an mp3 file
            asyncio.run(
                communicate.save(speech_file)
            )

            # Speak
            play_file(speech_file)

            # Small pause
            time.sleep(0.2)

            # Cry
            play_file("audio/cry.mp3")

        # -------------------------
        # KNOWN PERSON
        # -------------------------
        else:

            speech_file = "audio/speech.mp3"

            # Generate the spoken announcement text for a recognized face
            text = f"Mama, I see {name}!"

            # Use edge_tts with a slightly faster rate and even higher pitch for an excited tone
            communicate = edge_tts.Communicate(
                text,
                "en-US-AnaNeural",
                rate="-15%",
                pitch="+30Hz"
            )

            # Save the generated speech to an mp3 file
            asyncio.run(
                communicate.save(speech_file)
            )

            # Speak
            play_file(speech_file)

            # Small pause
            time.sleep(0.15)

            # Giggle
            play_file("audio/giggle.mp3")

    finally:
        # Always release the audio lock, even if an error occurred
        audio_playing = False



# Face detector
# Load OpenCV's pre-trained Haar Cascade classifier for frontal face detection
face_cascade = cv2.CascadeClassifier(
    os.path.join(
        os.path.dirname(__file__),
        "haarcascade_frontalface_default.xml"
    )
)

if face_cascade.empty():
    print("ERROR: Could not load face detector")
    exit()

# Open the default webcam (index 0)
camera = cv2.VideoCapture(0)

# Global flag used to prevent multiple audio threads from playing simultaneously
audio_playing = False

# Main camera loop — reads frames continuously until 'q' is pressed
while True:

    ret, frame = camera.read()

    # Convert frame to grayscale — both face detection and recognition work on grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    # scaleFactor=1.2: image is reduced by 20% at each scale step
    # minNeighbors=8: higher value = fewer false positives
    # minSize=(80,80): ignore faces smaller than 80×80 pixels
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=8,
        minSize=(80, 80)
    )


    for (x, y, w, h) in faces:

        # Crop the detected face region from the grayscale frame
        face = gray[y:y+h, x:x+w]

        # Predict the label and distance score for the cropped face
        # Lower distance = closer match (LBPH measures pixel pattern distance)
        label, distance = recognizer.predict(face)

        # Recognition
        # Distance < 80 means close enough to a known face to be a match
        if distance < 80:
            name = names.get(label, "Unknown")
        else:
            name = "Unknown"

        # Audio
        current_time = time.time()

        # Only announce once every 5 seconds to avoid spamming
        if name != "Unknown":

            if current_time - last_spoken >= 5:
                print(f"Mama, I see {name}!")
                speak_to_mama(name)
                last_spoken = current_time

        else:

            if current_time - last_spoken >= 5:
                print("Mama, I don't know who that is...")
                speak_to_mama("Unknown")
                last_spoken = current_time



        # Bounding box
        # Draw a green rectangle around the detected face
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Name tag
        # Overlay the person's name just above the bounding box
        cv2.putText(
            frame,
            name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )


    # Display the annotated frame in a window
    cv2.imshow("Mama's Camera", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
