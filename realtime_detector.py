import cv2
import numpy as np
import csv
import os
import time
import winsound
from datetime import datetime
import pyttsx3
from tensorflow.keras.models import load_model

# ── VOICE FEEDBACK ────────────────────────────────────────────────────────────
def speak(text: str):
    """Speak immediately for this detection; fallback to a beep on failure."""
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as exc:
        print(f"Audio error: {exc}")
        winsound.Beep(1500, 500)

def announce_result(result_label: str):
    if result_label == "NON-COMPLIANT":
        speak("Please follow proper dress code")
    else:
        speak("Thank you, you may enter")
# ──────────────────────────────────────────────────────────────────────────────

# ── REPORT SETUP ──────────────────────────────────────────────────────────────
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

SUMMARY_FILE = os.path.join(REPORTS_DIR, "daily_summary.csv")
_SUMMARY_HEADERS = ["Date", "Compliant", "Non-Compliant", "Total"]

def _ensure_summary_headers():
    if not os.path.isfile(SUMMARY_FILE) or os.path.getsize(SUMMARY_FILE) == 0:
        with open(SUMMARY_FILE, "w", newline="") as f:
            csv.writer(f).writerow(_SUMMARY_HEADERS)

_ensure_summary_headers()

def log_detection(result_label: str, raw_score: float, confidence: float):
    """Append one detection row to today's detail CSV and refresh daily summary."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # ── detail log (one row per detection) ────────────────────────────────────
    detail_file = os.path.join(REPORTS_DIR, f"detections_{date_str}.csv")
    detail_headers = ["Date", "Time", "Result", "Score", "Confidence"]
    write_header = not os.path.isfile(detail_file) or os.path.getsize(detail_file) == 0
    with open(detail_file, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(detail_headers)
        w.writerow([date_str, time_str, result_label, f"{raw_score:.4f}", f"{confidence:.2%}"])

    # ── rebuild today's totals from the detail file ───────────────────────────
    compliant = non_compliant = 0
    with open(detail_file, "r", newline="") as f:
        for row in csv.DictReader(f):
            if row["Result"] == "COMPLIANT":
                compliant += 1
            else:
                non_compliant += 1

    # ── update / insert today's row in the summary file ───────────────────────
    _update_summary(date_str, compliant, non_compliant)

def _update_summary(date_str, compliant, non_compliant):
    rows = []
    updated = False
    if os.path.isfile(SUMMARY_FILE):
        with open(SUMMARY_FILE, "r", newline="") as f:
            rows = list(csv.reader(f))
    new_row = [date_str, compliant, non_compliant, compliant + non_compliant]
    for i, row in enumerate(rows):
        if row and row[0] == date_str:
            rows[i] = new_row
            updated = True
            break
    if not updated:
        rows.append(new_row)
    with open(SUMMARY_FILE, "w", newline="") as f:
        csv.writer(f).writerows(rows)
# ──────────────────────────────────────────────────────────────────────────────

# 1. LOAD THE MODEL
# Ensure the filename matches exactly what is in your folder
model_path = "dress_code_detector (6).h5" 
model = load_model(model_path)

# Get input dimensions automatically from the model
_, img_h, img_w, _ = model.input_shape

# 2. CAMERA SETUP
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("--- Dress Code Detector Loaded ---")
print("Controls:")
print("  Auto-capture once after person stands for 5 seconds")
print("  SPACE -> capture immediately (optional override)")
print("  L     -> flip labels (if results seem reversed)")
print("  R     -> return to live view now")
print("  Q     -> quit")

RESULT_HOLD_SEC = 2.0
PERSON_STAND_SEC = 5.0
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

STATE_LIVE   = "live"
STATE_RESULT = "result"
state             = STATE_LIVE
result_frame      = None
result_frame_orig = None
score             = 0.0
labels_flipped    = False
result_shown_until = 0.0
captured_for_current_person = False
person_seen_since = None

def classify_frame(bgr_frame):
    # Convert BGR (OpenCV) to RGB (Model expectation)
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    
    # Resize to the size the model expects (224x224)
    sized = cv2.resize(rgb, (img_w, img_h))
    
    # PREPROCESSING: Match your Colab 'rescale=1./255'
    # We do NOT use mobilenet_v2.preprocess_input because your training used 1/255
    inp = sized.astype(np.float32) / 255.0
    
    # Add batch dimension
    inp = np.expand_dims(inp, axis=0)
    
    # Return the raw sigmoid score (0 to 1)
    prediction = model.predict(inp, verbose=0)
    return float(prediction[0][0])

def person_seen_in_frame(bgr_frame):
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    return len(faces) > 0

def draw_live_overlay(frame, person_seen, captured_for_person, standing_seconds_left):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 44), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    if person_seen and not captured_for_person:
        status = f"Person seen -> capture in {standing_seconds_left:0.1f}s"
    elif person_seen and captured_for_person:
        status = "Person still in frame (already captured)"
    else:
        status = "No person seen"
    cv2.putText(frame, f"{status}  SPACE=now  L=flip  Q=quit",
                (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    # Date & time in top-right corner
    dt_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    (tw, _), _ = cv2.getTextSize(dt_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, dt_str, (w - tw - 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

def draw_result_overlay(frame, score, flipped=False):
    # Determine class based on the 0.5 threshold
    # Default: > 0.5 is Non-Compliant (Alphabetical: Compliance=0, Non-Compliance=1)
    is_nc = (score > 0.5) if not flipped else (score <= 0.5)
    
    if is_nc:
        label, bar_color, text_color = "NON-COMPLIANT", (0, 0, 220), (0, 0, 255)
    else:
        label, bar_color, text_color = "COMPLIANT", (0, 180, 0), (0, 220, 0)

    # Calculate confidence percentage
    confidence = abs(score - 0.5) * 2
    h, w = frame.shape[:2]

    # Draw result UI
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, label,
                (14, 55), cv2.FONT_HERSHEY_DUPLEX, 1.6, text_color, 3)

    # Draw confidence bar
    bar_x, bar_y, bar_w, bar_h = 14, 68, w - 28, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    filled = int(bar_w * confidence)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), bar_color, -1)

    flip_note = "  [labels FLIPPED]" if flipped else ""
    cv2.putText(frame, f"Confidence: {confidence:.0%}  score={score:.3f}{flip_note}",
                (14, bar_y + bar_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    # Date & time stamp on result screen
    dt_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, dt_str,
                (14, bar_y + bar_h + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(frame, "R=retake  L=flip labels  Q=quit",
                (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

# 3. MAIN LOOP
while True:
    key = cv2.waitKey(1)

    if key == ord("q") or key == ord("Q"):
        break

    # Toggle logic if classes are reversed
    if key == ord("l") or key == ord("L"):
        labels_flipped = not labels_flipped
        print(f"Label flip: {labels_flipped}")
        if state == STATE_RESULT and result_frame_orig is not None:
            result_frame = result_frame_orig.copy()
            draw_result_overlay(result_frame, score, flipped=labels_flipped)

    if state == STATE_LIVE:
        ret, frame = cap.read()
        if not ret:
            break

        person_seen = person_seen_in_frame(frame)
        if not person_seen:
            person_seen_since = None
            captured_for_current_person = False
        elif person_seen_since is None:
            person_seen_since = time.monotonic()

        standing_elapsed = 0.0 if person_seen_since is None else (time.monotonic() - person_seen_since)
        standing_seconds_left = max(0.0, PERSON_STAND_SEC - standing_elapsed)

        draw_live_overlay(frame, person_seen, captured_for_current_person, standing_seconds_left)
        cv2.imshow("Dress Code Detector", frame)

        auto_capture_due = person_seen and (standing_elapsed >= PERSON_STAND_SEC) and not captured_for_current_person
        if key == ord(" ") or auto_capture_due: # Manual capture or one-time per person
            captured = frame.copy()
            result_frame_orig = captured.copy()
            print("Classifying...", end=" ", flush=True)
            
            score = classify_frame(captured)

            is_nc = (score > 0.5) if not labels_flipped else (score <= 0.5)
            result_label = "NON-COMPLIANT" if is_nc else "COMPLIANT"
            confidence = abs(score - 0.5) * 2
            log_detection(result_label, score, confidence)
            print(f"Logged: {result_label} (score={score:.4f}, conf={confidence:.0%})")

            result_frame = result_frame_orig.copy()
            draw_result_overlay(result_frame, score, flipped=labels_flipped)
            # Show result first, then announce audio.
            cv2.imshow("Dress Code Detector", result_frame)
            cv2.waitKey(1)

            # Voice feedback for every detection
            announce_result(result_label)

            state = STATE_RESULT
            result_shown_until = time.monotonic() + RESULT_HOLD_SEC
            if person_seen:
                captured_for_current_person = True
            print("Done.")

    elif state == STATE_RESULT:
        cv2.imshow("Dress Code Detector", result_frame)
        if key == ord("r") or key == ord("R"):
            state = STATE_LIVE
        elif time.monotonic() >= result_shown_until:
            state = STATE_LIVE

cap.release()
cv2.destroyAllWindows()