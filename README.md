# 🖐 Hand Gesture Volume Controller

Control the volume of **any audio device** — including Bluetooth headphones — using hand gestures detected through your webcam. No clicks, no keyboard shortcuts, just pinch and spread.

---

## ✋ How It Works

| Gesture | Effect |
|---|---|
| **Pinch** thumb + index finger close together | Volume → 0% |
| **Spread** thumb + index finger far apart | Volume → 100% |
| Hand not visible | Volume stays at last value |
| Press **Q** or **ESC** | Quit |

On startup you'll see a list of all your audio output devices (including Bluetooth). Just type the number of the one you want to control.

```
🔊  Available audio output devices:
──────────────────────────────────────────────────────
  [0]  Headphones (Airdopes 141)        (bluetooth/unplugged)
  [1]  Speaker (Realtek(R) Audio)       (active)  ← default
  [2]  Headphones (SoundDrum 1)         (active)
  [3]  Headphones (Rockerz 245 V2 Pro)  (bluetooth/unplugged)
──────────────────────────────────────────────────────
  Enter number [0–3] (or press Enter for default):
```

---

## 🖥️ Requirements

| Requirement | Detail |
|---|---|
| **OS** | Windows 10 / 11 only (pycaw is Windows-exclusive) |
| **Python** | 3.11 or newer |
| **Webcam** | Any USB or built-in webcam |
| **Internet** | Needed once on first run to download the MediaPipe model (~25 MB) |

---

## 📦 Installation

```bash
# Install all dependencies
pip install opencv-python mediapipe numpy pycaw

# Run the app
python volume_controller.py
```

---

## 📁 Project Files

```
hand_gesture_volume_controller/
├── volume_controller.py   ← main application
├── debug_devices.py       ← run this to list all audio devices on your system
├── hand_landmarker.task   ← auto-downloaded on first run (~25 MB)
└── README.md
```

---

## ⚙️ Calibration

If the volume feels too jumpy or barely moves, open `volume_controller.py` and adjust these two values near the top to match your hand size and how far you sit from the camera:

```python
DIST_MIN: float = 30.0   # pixel distance between fingers = 0% volume
DIST_MAX: float = 220.0  # pixel distance between fingers = 100% volume
```

**How to find your values:**
1. Run the app and watch the volume bar.
2. Fully pinch your fingers — if volume doesn't reach 0%, lower `DIST_MIN`.
3. Fully spread your fingers — if volume doesn't reach 100%, raise `DIST_MAX`.

---

## 🔵 Bluetooth Device Notes

Windows reports Bluetooth audio devices as `Unplugged` in the audio API even when they are actively connected and playing — this is a known Windows quirk, not a bug in this app.

The device picker includes `Unplugged` devices specifically to support Bluetooth. If your Bluetooth headphones don't appear in the list:

1. Make sure they are **paired and connected** in Windows Bluetooth settings.
2. Check that they appear in **Windows Sound Settings → Output devices**.
3. Run `debug_devices.py` to see the raw device list and their states.

---

## 🏗️ Code Architecture

```
Startup
  └── pick_audio_device()
        └── list_output_devices()   filters Active + Unplugged, excludes mics/HFP

Main Loop (per frame)
  ├── cv2.VideoCapture            grab webcam frame
  ├── cv2.flip()                  mirror so it feels natural
  ├── detect_hand()               MediaPipe HandLandmarker (Tasks API)
  ├── get_fingertip_coords()      landmark 4 (thumb) + landmark 8 (index) → pixels
  ├── distance_to_volume_pct()    Euclidean distance → deque smoothing → 0–100%
  ├── set_device_volume()         pycaw EndpointVolume.SetMasterVolumeLevel()
  ├── draw_hand_skeleton()        manual OpenCV skeleton (21 joints + bones)
  └── draw_overlays()             volume bar, FPS, device name, instructions
```

---

## 🐛 Troubleshooting

| Problem | Fix |
|---|---|
| `MediaPipe Tasks API error` | Run `pip install --upgrade mediapipe` |
| `No usable output devices found` | Run `debug_devices.py` and paste the output for diagnosis |
| Bluetooth device not in list | Make sure it's connected in Windows Bluetooth Settings |
| Volume jumps around | Increase `SMOOTHING_WINDOW` (default: 20) in the config section |
| Low FPS | Lower `FRAME_WIDTH`/`FRAME_HEIGHT` to `640`/`480` in config |
| Camera not found | Change `CAMERA_INDEX` from `0` to `1` or `2` |
| Model download fails | Manually download from the URL in `MODEL_URL` and save as `hand_landmarker.task` |

---

## 📚 Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Webcam capture and all drawing/overlays |
| `mediapipe` | Hand landmark detection (21 points per hand) |
| `numpy` | Distance mapping, interpolation, smoothing |
| `pycaw` | Windows audio endpoint volume control |
