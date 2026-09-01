# Peek-a-Boo

A baby peek-a-boo game built with OpenCV. The camera watches for faces, recognizes who it sees, and announces them in a child's voice — giggling when it spots a familiar face and crying when it sees a stranger.

## How it works

1. **Capture** — run `face_detection.py` to photograph each person via webcam. Images are saved to `faces/<name>/` at 2-second intervals whenever a face is detected.
2. **Train** — run `train.py` to build an LBPH face recognition model from those photos. Outputs `faces.yml` (the model) and `names.txt` (the label map).
3. **Play** — run `recognize.py` to start the live game. When a face appears on camera:
   - Known face → announces *"Mama, I see \<name\>!"* in a child's voice, then giggles
   - Unknown face → announces *"Mamaaa... I don't know who that is..."* then cries

Audio is synthesized on the fly using [edge-tts](https://github.com/rany2/edge-tts) with a child-like voice (en-US-AnaNeural at reduced rate and raised pitch), then played back via the system audio player. Announcements are throttled to once every 5 seconds to avoid repeating.

## Requirements

- Python 3.8+
- OpenCV with the `opencv-contrib-python` package (for LBPH recognizer)
- edge-tts
- A working webcam

```
pip install opencv-contrib-python edge-tts miniaudio
```

Works on macOS, Windows, and Linux. `miniaudio` bundles its own audio library with no external dependencies, so it works regardless of SDL or system audio setup.

## Usage

### Step 1 — Capture face photos

```bash
python face_detection.py
```

Enter a name when prompted. The script opens your webcam and saves a photo every 2 seconds whenever it detects a face. Collect 10–20 images per person for best results. Press `q` to stop.

Repeat for each person you want the game to recognize.

### Step 2 — Train the model

```bash
python train.py
```

Reads all images from `faces/`, trains the recognizer, and writes `faces.yml` and `names.txt`.

### Step 3 — Play peek-a-boo

```bash
python recognize.py
```

Opens the webcam and starts announcing faces. Press `q` to quit.

## Project structure

```
.
├── face_detection.py            # Step 1: capture training photos
├── train.py                     # Step 2: train the face recognizer
├── recognize.py                 # Step 3: live peek-a-boo game
├── haarcascade_frontalface_default.xml   # Haar Cascade face detector
│
│   # generated — excluded from git
├── faces/                       # captured training images (one folder per person)
├── faces.yml                    # trained LBPH model
├── names.txt                    # label → name mapping
└── audio/                       # synthesized and effect audio files
```

## Notes
- No biometric data is committed to git — see `.gitignore`.
