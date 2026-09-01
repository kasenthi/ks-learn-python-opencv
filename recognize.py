# Face recognition camera that announces recognized people via text-to-speech.
# Uses a pre-trained LBPH model (faces.yml) and speaks using the macOS `afplay` command.
# Designed to alert "Mama" when a known or unknown face is detected on camera.

import cv2
import os
import sys
import asyncio
import edge_tts
import time
import threading
import miniaudio

def play_file(path):
    done = threading.Event()

    def guarded_stream():
        yield from miniaudio.stream_file(path)
        done.set()

    gen = guarded_stream()
    next(gen)  # prime the generator before miniaudio calls send(framecount)

    with miniaudio.PlaybackDevice() as device:
        device.start(gen)
        done.wait()

# Tracks the last name spoken and when it was spoken, to avoid repeating announcements
last_name = ""
last_spoken = 0

# Load the pre-trained LBPH face recognizer model if available
model_trained = os.path.exists("faces.yml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
if model_trained:
    recognizer.read("faces.yml")
else:
    print("No trained model found (faces.yml missing). All faces will be treated as unknown.")

# Load names if available
names = {}
if model_trained and os.path.exists("names.txt"):
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

stop_event = threading.Event()

def terminal_watcher():
    print("Press 'q' + Enter in the terminal, or 'q' in the camera window to quit.")
    for line in sys.stdin:
        if line.strip().lower() == "q":
            stop_event.set()
            return

threading.Thread(target=terminal_watcher, daemon=True).start()

# Main camera loop
while not stop_event.is_set():

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

        # Skip prediction if no model is trained — treat everyone as unknown
        if not model_trained:
            name = "Unknown"
        else:
            label, distance = recognizer.predict(face)
            name = names.get(label, "Unknown") if distance < 80 else "Unknown"

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

    # Exit on 'q' in the camera window, or when the window is closed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or cv2.getWindowProperty("Mama's Camera", cv2.WND_PROP_VISIBLE) < 1:
        stop_event.set()

# Release the webcam and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
