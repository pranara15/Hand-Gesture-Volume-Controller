import sys
import os
import time
import urllib.request
from collections import deque
from math import hypot

try:
    import cv2
except ImportError:
    sys.exit("❌  OpenCV not found.  Run:  pip install opencv-python")

try:
    import numpy as np
except ImportError:
    sys.exit("❌  NumPy not found.  Run:  pip install numpy")

try:
    import mediapipe as mp          # type: ignore
    import mediapipe.tasks as mt    # type: ignore
    _vision               = mt.vision
    BaseOptions           = mt.BaseOptions
    HandLandmarker        = _vision.HandLandmarker
    HandLandmarkerOptions = _vision.HandLandmarkerOptions
    RunningMode           = _vision.RunningMode
except ImportError:
    sys.exit("❌  MediaPipe not found.  Run:  pip install mediapipe")
except AttributeError as e:
    sys.exit(f"❌  MediaPipe Tasks API error: {e}\n    Run:  pip install --upgrade mediapipe")

try:
    from pycaw.pycaw import AudioUtilities  # type: ignore
except ImportError:
    sys.exit("❌  pycaw not found.  Run:  pip install pycaw")


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CAMERA_INDEX:     int   = 0
FRAME_WIDTH:      int   = 1280
FRAME_HEIGHT:     int   = 720
DIST_MIN:         float = 30.0
DIST_MAX:         float = 220.0
SMOOTHING_WINDOW: int   = 20

MODEL_PATH: str = "hand_landmarker.task"
MODEL_URL:  str = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

COLOR_LINE     = (0,   255, 255)
COLOR_THUMB    = (255, 100,   0)
COLOR_INDEX    = (0,   100, 255)
COLOR_BAR_BG   = (50,   50,  50)
COLOR_BAR_FILL = (0,   255, 120)
COLOR_BAR_LOW  = (0,    80, 255)
COLOR_TEXT     = (255, 255, 255)
COLOR_JOINT    = (0,   255, 120)
COLOR_BONE     = (255, 255,   0)
FONT           = cv2.FONT_HERSHEY_SIMPLEX

BAR_X:      int = 40
BAR_Y_TOP:  int = 150
BAR_WIDTH:  int = 30
BAR_HEIGHT: int = 300

THUMB_TIP_ID = 4
INDEX_TIP_ID = 8

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

EXCLUDE_KEYWORDS = ("Hands-Free", "Microphone", "Stereo Mix", "Microphone Array")

from pycaw.pycaw import AudioDeviceState  # type: ignore
ACCEPTED_STATES = {AudioDeviceState.Active, AudioDeviceState.Unplugged}


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Audio device selection
# ══════════════════════════════════════════════════════════════════════════════

def list_output_devices() -> list[tuple]:
    
    all_devices = AudioUtilities.GetAllDevices()
    output = []
    seen   = set()   # deduplicate by FriendlyName

    for dev in all_devices:
        name = dev.FriendlyName

        # Skip duplicates (pycaw can list the same device twice)
        if name in seen:
            continue

        # Skip input devices and Hands-Free (HFP) Bluetooth profiles
        if any(kw.lower() in name.lower() for kw in EXCLUDE_KEYWORDS):
            continue

        # Skip states we can't use (NotPresent, Disabled)
        if dev.state not in ACCEPTED_STATES:
            continue

        # Skip devices with no volume interface
        try:
            ep = dev.EndpointVolume
            if ep is None:
                continue
            ep.GetVolumeRange()   # will raise if truly inaccessible
        except Exception:
            continue

        seen.add(name)
        output.append((dev, ep))

    return output


def pick_audio_device() -> tuple:
    """
    Show the filtered device list and let the user pick one.
    Returns (AudioDevice, EndpointVolume).
    """
    devices = list_output_devices()

    if not devices:
        sys.exit(
            "❌  No usable output devices found.\n"
            "    Make sure your Bluetooth device is paired and connected in Windows."
        )

    try:
        default_name = AudioUtilities.GetSpeakers().FriendlyName
    except Exception:
        default_name = ""

    print("\n🔊  Available audio output devices:")
    print("─" * 54)
    for i, (dev, _) in enumerate(devices):
        state_str = "active" if str(dev.state) == "AudioDeviceState.Active" else "bluetooth/unplugged"
        marker    = "  ← default" if dev.FriendlyName == default_name else ""
        print(f"  [{i}]  {dev.FriendlyName}  ({state_str}){marker}")
    print("─" * 54)

    if len(devices) == 1:
        dev, ep = devices[0]
        print(f"  Only one device found — using: {dev.FriendlyName}\n")
        return dev, ep

    while True:
        try:
            raw = input(
                f"  Enter number [0–{len(devices)-1}]"
                "  (or press Enter for default): "
            ).strip()

            if raw == "":
                for dev, ep in devices:
                    if dev.FriendlyName == default_name:
                        print(f"  Using default: {dev.FriendlyName}\n")
                        return dev, ep
                dev, ep = devices[0]
                print(f"  Using: {dev.FriendlyName}\n")
                return dev, ep

            idx = int(raw)
            if 0 <= idx < len(devices):
                dev, ep = devices[idx]
                print(f"  ✅  Selected: {dev.FriendlyName}\n")
                return dev, ep
            print(f"  ⚠️  Enter a number between 0 and {len(devices)-1}.")
        except ValueError:
            print("  ⚠️  Invalid — enter a number or press Enter.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — MediaPipe model + detector
# ══════════════════════════════════════════════════════════════════════════════

def download_model_if_needed(path: str = MODEL_PATH, url: str = MODEL_URL) -> str:
    if os.path.exists(path):
        print(f"✅  Model found: {path}")
        return path
    print("📥  Downloading hand landmarker model (~25 MB) — one-time setup…")
    try:
        urllib.request.urlretrieve(url, path)
        print(f"✅  Model saved to: {path}")
    except Exception as exc:
        if os.path.exists(path):
            os.remove(path)
        sys.exit(f"❌  Download failed: {exc}\n    URL: {url}")
    return path


def build_hand_detector(model_path: str):
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return HandLandmarker.create_from_options(options)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Open webcam
# ══════════════════════════════════════════════════════════════════════════════

def open_camera(index: int = CAMERA_INDEX) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera at index {index}. "
            "Check it is connected and not used by another app."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4-5 — Hand detection + distance → volume %
# ══════════════════════════════════════════════════════════════════════════════

def detect_hand(detector, rgb_frame: np.ndarray):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return detector.detect(mp_image)


def get_fingertip_coords(result, frame_w: int, frame_h: int) -> tuple:
    lms   = result.hand_landmarks[0]
    thumb = (int(lms[THUMB_TIP_ID].x * frame_w), int(lms[THUMB_TIP_ID].y * frame_h))
    index = (int(lms[INDEX_TIP_ID].x * frame_w), int(lms[INDEX_TIP_ID].y * frame_h))
    return thumb, index


def distance_to_volume_pct(thumb: tuple, index: tuple, history: deque) -> tuple:
    raw_dist = hypot(index[0] - thumb[0], index[1] - thumb[1])
    history.append(raw_dist)
    smoothed = sum(history) / len(history)
    vol_pct  = float(np.clip(
        np.interp(smoothed, [DIST_MIN, DIST_MAX], [0.0, 100.0]), 0.0, 100.0
    ))
    return raw_dist, vol_pct


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Apply volume to chosen device
# ══════════════════════════════════════════════════════════════════════════════

def set_device_volume(ep, vol_pct: float, vol_min: float, vol_max: float) -> None:
    target_db = float(np.interp(vol_pct, [0.0, 100.0], [vol_min, vol_max]))
    ep.SetMasterVolumeLevel(target_db, None)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — Draw hand skeleton
# ══════════════════════════════════════════════════════════════════════════════

def draw_hand_skeleton(frame: np.ndarray, result, frame_w: int, frame_h: int) -> None:
    lms = result.hand_landmarks[0]
    pts = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in lms]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], COLOR_BONE, 2, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, 5, COLOR_JOINT, cv2.FILLED)
        cv2.circle(frame, pt, 5, (0, 0, 0), 1)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8 — UI overlays
# ══════════════════════════════════════════════════════════════════════════════

def draw_overlays(
    frame: np.ndarray,
    thumb: tuple,
    index: tuple,
    vol_pct: float,
    fps: float,
    hand_detected: bool,
    device_name: str,
) -> None:
    h, w = frame.shape[:2]

    if hand_detected:
        cv2.line(frame, thumb, index, COLOR_LINE, 3, cv2.LINE_AA)
        mid = ((thumb[0] + index[0]) // 2, (thumb[1] + index[1]) // 2)
        cv2.circle(frame, mid, 8, COLOR_LINE, cv2.FILLED)
        cv2.circle(frame, thumb, 14, COLOR_THUMB, cv2.FILLED)
        cv2.circle(frame, thumb, 14, (255, 255, 255), 2)
        cv2.circle(frame, index, 14, COLOR_INDEX, cv2.FILLED)
        cv2.circle(frame, index, 14, (255, 255, 255), 2)

    bar_bottom = BAR_Y_TOP + BAR_HEIGHT
    cv2.rectangle(frame, (BAR_X, BAR_Y_TOP),
                  (BAR_X + BAR_WIDTH, bar_bottom), COLOR_BAR_BG, cv2.FILLED)
    fill_top  = int(bar_bottom - (vol_pct / 100.0) * BAR_HEIGHT)
    bar_color = COLOR_BAR_LOW if vol_pct < 20 else COLOR_BAR_FILL
    if fill_top < bar_bottom:
        cv2.rectangle(frame, (BAR_X, fill_top),
                      (BAR_X + BAR_WIDTH, bar_bottom), bar_color, cv2.FILLED)
    cv2.rectangle(frame, (BAR_X, BAR_Y_TOP),
                  (BAR_X + BAR_WIDTH, bar_bottom), (200, 200, 200), 2)
    cv2.putText(frame, "VOL",  (BAR_X - 2, BAR_Y_TOP - 15),
                FONT, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{int(vol_pct)}%", (BAR_X - 2, bar_bottom + 28),
                FONT, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)

    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 130, 36),
                FONT, 0.7, (180, 255, 100), 2, cv2.LINE_AA)

    label = f"Device: {device_name}"
    tsz   = cv2.getTextSize(label, FONT, 0.52, 1)[0]
    tx    = (w - tsz[0]) // 2
    cv2.rectangle(frame, (tx - 8, 8), (tx + tsz[0] + 8, 32), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, label, (tx, 26), FONT, 0.52, (180, 220, 255), 1, cv2.LINE_AA)

    if not hand_detected:
        msg = "Show your hand to control volume"
        msz = cv2.getTextSize(msg, FONT, 0.8, 2)[0]
        mx  = (w - msz[0]) // 2
        cv2.rectangle(frame, (mx - 14, h//2 - 34),
                      (mx + msz[0] + 14, h//2 + 14), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, msg, (mx, h // 2),
                    FONT, 0.8, (100, 200, 255), 2, cv2.LINE_AA)

    cv2.rectangle(frame, (0, h - 36), (w, h), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, "Pinch = vol down  |  Spread = vol up  |  Q to quit",
                (14, h - 10), FONT, 0.55, (180, 180, 180), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("🖐  Hand Gesture Volume Controller")
    print("   Press  Q  or  ESC  to quit.\n")

    selected_dev, ep = pick_audio_device()
    device_name      = selected_dev.FriendlyName
    vol_range        = ep.GetVolumeRange()
    vol_min, vol_max = vol_range[0], vol_range[1]
    print(f"🔊  Controlling: {device_name}  ({vol_min:.1f} dB → {vol_max:.1f} dB)")

    model_path = download_model_if_needed()
    detector   = build_hand_detector(model_path)
    print("🤚  Hand landmarker ready.")

    try:
        cap = open_camera(CAMERA_INDEX)
        print(f"📷  Camera opened (index {CAMERA_INDEX}).\n")
    except RuntimeError as exc:
        sys.exit(f"❌  {exc}")

    dist_history: deque = deque(maxlen=SMOOTHING_WINDOW)
    vol_pct   = 50.0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame            = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]
        rgb_frame        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result           = detect_hand(detector, rgb_frame)

        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        hand_detected       = False
        thumb_pt = index_pt = (0, 0)

        if result.hand_landmarks:
            hand_detected      = True
            thumb_pt, index_pt = get_fingertip_coords(result, frame_w, frame_h)
            _, vol_pct         = distance_to_volume_pct(thumb_pt, index_pt, dist_history)
            set_device_volume(ep, vol_pct, vol_min, vol_max)
            draw_hand_skeleton(frame, result, frame_w, frame_h)

        draw_overlays(frame, thumb_pt, index_pt, vol_pct, fps, hand_detected, device_name)
        cv2.imshow("Hand Gesture Volume Controller", frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            print("\n👋  Exiting…")
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("✅  Done. Goodbye!")


if __name__ == "__main__":
    main()
