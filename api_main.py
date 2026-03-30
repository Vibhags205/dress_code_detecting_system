import os
import csv
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

import cv2
import numpy as np
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from tensorflow.keras.models import load_model


MODEL_PATH = os.getenv("MODEL_PATH", "dress_code_detector (6).h5")
THRESHOLD = float(os.getenv("DRESS_THRESHOLD", "0.5"))
PHONE_THRESHOLD = float(os.getenv("DRESS_THRESHOLD_PHONE", str(THRESHOLD)))
LAPTOP_THRESHOLD = float(os.getenv("DRESS_THRESHOLD_LAPTOP", str(THRESHOLD)))
REPORTS_DIR = os.getenv("REPORTS_DIR", "reports")
SUMMARY_FILE = os.path.join(REPORTS_DIR, "daily_summary.csv")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8300038302:AAFVG5i_ve2SwMgsjPuPGqHFYJXtAb4YYzs")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "-1003871574876")
TELEGRAM_ALERT_COOLDOWN_SEC = int(os.getenv("TELEGRAM_ALERT_COOLDOWN_SEC", "30"))

app = FastAPI(title="Dress Code Detector API", version="1.0.0")

model = None
img_h = 224
img_w = 224
inference_lock = threading.Lock()
report_lock = threading.Lock()
last_telegram_alert_at = 0.0


def ensure_report_files() -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    if not os.path.isfile(SUMMARY_FILE) or os.path.getsize(SUMMARY_FILE) == 0:
        with open(SUMMARY_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["Date", "Compliant", "Non-Compliant", "Total"])


def log_detection(result: str, score: float, confidence: float) -> Dict[str, Any]:
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    detail_file = os.path.join(REPORTS_DIR, f"detections_{date_str}.csv")
    write_header = not os.path.isfile(detail_file) or os.path.getsize(detail_file) == 0

    with report_lock:
        with open(detail_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Date", "Time", "Result", "Score", "Confidence"])
            writer.writerow([date_str, time_str, result, f"{score:.4f}", f"{confidence:.2%}"])

        compliant = 0
        non_compliant = 0
        with open(detail_file, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("Result") == "COMPLIANT":
                    compliant += 1
                elif row.get("Result") == "NON-COMPLIANT":
                    non_compliant += 1

        rows = []
        if os.path.isfile(SUMMARY_FILE):
            with open(SUMMARY_FILE, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))

        updated = False
        new_row = [date_str, compliant, non_compliant, compliant + non_compliant]
        for i, row in enumerate(rows):
            if row and row[0] == date_str:
                rows[i] = new_row
                updated = True
                break
        if not updated:
            rows.append(new_row)

        with open(SUMMARY_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)

    return {
        "date": date_str,
        "time": time_str,
        "detail_file": detail_file,
        "summary_file": SUMMARY_FILE,
    }


def read_csv_rows(file_path: str, limit: int = 200) -> List[Dict[str, Any]]:
    if not os.path.isfile(file_path):
        return []

    with open(file_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit <= 0:
        return rows
    return rows[-limit:]


def maybe_send_telegram_alert(image_bgr: np.ndarray, result: str, confidence: float) -> bool:
    global last_telegram_alert_at

    if result != "NON-COMPLIANT":
        return False
    
    if not TELEGRAM_BOT_TOKEN:
        print("[Telegram] Skipped: BOT_TOKEN not set")
        return False
    if not TELEGRAM_CHAT_ID:
        print("[Telegram] Skipped: CHAT_ID not set")
        return False

    now_ts = time.time()
    if (now_ts - last_telegram_alert_at) < TELEGRAM_ALERT_COOLDOWN_SEC:
        print(f"[Telegram] Skipped: cooldown active ({now_ts - last_telegram_alert_at:.1f}s < {TELEGRAM_ALERT_COOLDOWN_SEC}s)")
        return False

    ok, buffer = cv2.imencode(".jpg", image_bgr)
    if not ok:
        print("[Telegram] Skipped: image encode failed")
        return False

    message = (
        "Dress code violation detected\n\n"
        f"Date: {datetime.now().strftime('%Y-%m-%d')}\n"
        f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
        f"Confidence: {confidence:.0%}"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {"photo": ("alert.jpg", buffer.tobytes(), "image/jpeg")}
    data = {"chat_id": TELEGRAM_CHAT_ID, "caption": message}

    try:
        print(f"[Telegram] Sending alert to chat {TELEGRAM_CHAT_ID}...")
        resp = requests.post(url, data=data, files=files, timeout=12)
        if resp.ok:
            print(f"[Telegram] Success: {resp.status_code}")
            last_telegram_alert_at = now_ts
            return True
        else:
            print(f"[Telegram] Failed: {resp.status_code} {resp.text}")
            return False
    except Exception as e:
        print(f"[Telegram] Error: {e}")
        return False


def resolve_model_path() -> str:
    # Prefer explicit MODEL_PATH, then common filenames, then any .h5 in project root.
    candidates = [
        Path(MODEL_PATH),
        Path("dress_code_detector (6).h5"),
        Path("dress_code_detector.h5"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    any_h5 = sorted(Path(".").glob("*.h5"))
    if any_h5:
        return str(any_h5[0])

    raise RuntimeError(
        "Model file not found. Set MODEL_PATH env var or include a .h5 model file in the app root."
    )


@app.on_event("startup")
def load_detection_model() -> None:
    global model, img_h, img_w

    ensure_report_files()
    model_file = resolve_model_path()
    model = load_model(model_file)
    _, img_h, img_w, _ = model.input_shape


def _resolve_threshold(device_type: str) -> float:
    device = (device_type or "").strip().lower()
    if device == "phone":
        return PHONE_THRESHOLD
    if device == "laptop":
        return LAPTOP_THRESHOLD
    return THRESHOLD


def classify_image(bgr_image: np.ndarray, device_type: str = "unknown") -> Dict[str, Any]:
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # Use three views to stabilize predictions under phone/laptop camera noise.
    full = cv2.resize(rgb, (img_w, img_h))
    h, w = rgb.shape[:2]
    crop_h = max(1, int(h * 0.9))
    crop_w = max(1, int(w * 0.9))
    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2
    center_crop = cv2.resize(rgb[y0:y0 + crop_h, x0:x0 + crop_w], (img_w, img_h))
    flipped = cv2.flip(full, 1)

    inp = np.stack([full, center_crop, flipped], axis=0).astype(np.float32) / 255.0

    with inference_lock:
        prediction = model.predict(inp, verbose=0)

    score = float(np.mean(prediction[:, 0]))
    threshold = _resolve_threshold(device_type)
    is_non_compliant = score > threshold
    result = "NON-COMPLIANT" if is_non_compliant else "COMPLIANT"
    confidence = abs(score - threshold) * (1.0 / max(threshold, 1.0 - threshold))
    confidence = float(max(0.0, min(confidence, 1.0)))

    return {
        "result": result,
        "score": round(score, 6),
        "confidence": round(confidence, 6),
        "threshold": threshold,
        "device_type": device_type,
    }


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Dress Code Live Detector</title>
    <style>
        :root {
            --bg: #f4f6ef;
            --panel: #ffffff;
            --text: #102218;
            --muted: #53615a;
            --accent: #1f7a4d;
            --bad: #b42318;
            --border: #d8e0db;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: "Segoe UI", "Trebuchet MS", sans-serif;
            background: radial-gradient(circle at top right, #e9f7e7, var(--bg));
            color: var(--text);
            min-height: 100vh;
            padding: 20px;
        }
        .wrap {
            max-width: 900px;
            margin: 0 auto;
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 18px;
            box-shadow: 0 10px 30px rgba(13, 35, 24, 0.08);
        }
        h1 { margin: 0 0 6px; font-size: 1.35rem; }
        p { margin: 0 0 12px; color: var(--muted); }
        video {
            width: 100%;
            border-radius: 12px;
            background: #111;
            border: 1px solid var(--border);
            aspect-ratio: 16 / 9;
            object-fit: cover;
        }
        .row {
            margin-top: 14px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            align-items: center;
        }
        button {
            border: 0;
            padding: 10px 14px;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
            background: var(--accent);
            color: #fff;
        }
        button.secondary { background: #475850; }
        label { color: var(--muted); font-size: 0.95rem; }
        input[type=number] {
            width: 90px;
            padding: 8px;
            border-radius: 8px;
            border: 1px solid var(--border);
        }
        .result {
            margin-top: 14px;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 12px;
            background: #fafcf9;
        }
        .badge {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            font-weight: 700;
            background: #e8f5ed;
            color: #145a38;
        }
        .badge.bad { background: #fdeceb; color: var(--bad); }
        .small { color: var(--muted); font-size: 0.9rem; }
        .status-bar { 
            margin-top: 8px; 
            padding: 8px; 
            background: #f0f2ed; 
            border-radius: 8px; 
            font-size: 0.85rem; 
        }
        .status-bar.person-found { background: #d4edda; }
    </style>
</head>
<body>
    <div class="wrap">
        <h1>Dress Code Live Detector</h1>
        <p>Open this link on any phone or PC, allow camera access. Auto-captures every 5s when person detected.</p>
        <video id="video" playsinline autoplay muted></video>
        <canvas id="canvas" width="640" height="360" style="display:none"></canvas>

        <div class="row">
            <button id="startBtn">Start Live Detection</button>
            <button id="stopBtn" class="secondary">Stop</button>
        </div>

        <div class="status-bar" id="statusBar">
            <div class="small">Ready to start.</div>
        </div>

        <div class="result" id="resultBox">
            <div class="small">No detections yet.</div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd"></script>
    <script>
        const video = document.getElementById("video");
        const canvas = document.getElementById("canvas");
        const resultBox = document.getElementById("resultBox");
        const statusBar = document.getElementById("statusBar");
        const startBtn = document.getElementById("startBtn");
        const stopBtn = document.getElementById("stopBtn");

        let stream = null;
        let timer = null;
        let model = null;
        let isRunning = false;
        let personDetectedSince = null;
        let lastCaptureTime = null;
        let deviceType = "laptop";

        function detectDeviceType() {
            const ua = navigator.userAgent || "";
            const mobileByUA = /Android|iPhone|iPad|iPod|Mobile/i.test(ua);
            const touchCapable = navigator.maxTouchPoints && navigator.maxTouchPoints > 1;
            return (mobileByUA || touchCapable) ? "phone" : "laptop";
        }

        async function loadModel() {
            statusBar.innerHTML = `<div class="small">Loading object detection model...</div>`;
            model = await cocoSsd.load();
            statusBar.innerHTML = `<div class="small">Model loaded. Ready to start.</div>`;
        }

        async function startCamera() {
            if (stream) return;

            deviceType = detectDeviceType();
            const baseConstraints = deviceType === "phone"
                ? {
                    facingMode: { ideal: "environment" },
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    frameRate: { ideal: 24, max: 30 },
                }
                : {
                    facingMode: { ideal: "user" },
                    width: { ideal: 960 },
                    height: { ideal: 540 },
                    frameRate: { ideal: 20, max: 30 },
                };

            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: baseConstraints,
                    audio: false,
                });
            } catch {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: true,
                    audio: false,
                });
            }

            const [track] = stream.getVideoTracks();
            if (track) {
                const advanced = deviceType === "phone"
                    ? [{ focusMode: "continuous" }, { exposureMode: "continuous" }, { whiteBalanceMode: "continuous" }]
                    : [{ exposureMode: "continuous" }, { whiteBalanceMode: "continuous" }];
                try {
                    await track.applyConstraints({ advanced });
                } catch {
                    // Not all browsers/devices support advanced camera controls.
                }
            }

            video.srcObject = stream;
            statusBar.className = "status-bar";
            statusBar.innerHTML = `<div class="small">Camera started (${deviceType}).</div>`;
        }

        function stopCamera() {
            if (!stream) return;
            stream.getTracks().forEach(t => t.stop());
            stream = null;
        }

        async function detectPersonInFrame() {
            if (!video.videoWidth || !video.videoHeight || !model) return false;
            const predictions = await model.detect(video);
            return predictions.some(p => p.class === "person" && p.score > 0.5);
        }

        async function getBestPersonPrediction() {
            if (!video.videoWidth || !video.videoHeight || !model) return null;
            const predictions = await model.detect(video);
            const people = predictions.filter(p => p.class === "person" && p.score > 0.5);
            if (!people.length) return null;
            people.sort((a, b) => b.score - a.score);
            return people[0];
        }

        async function detectOnce() {
            if (!video.videoWidth || !video.videoHeight) return;
            const ctx = canvas.getContext("2d");
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            const person = await getBestPersonPrediction();
            let sourceCanvas = canvas;

            if (person && person.bbox && person.bbox.length === 4) {
                const [x, y, w, h] = person.bbox;
                const padX = w * 0.2;
                const padY = h * 0.2;
                const sx = Math.max(0, Math.floor(x - padX));
                const sy = Math.max(0, Math.floor(y - padY));
                const ex = Math.min(canvas.width, Math.ceil(x + w + padX));
                const ey = Math.min(canvas.height, Math.ceil(y + h + padY));
                const sw = Math.max(1, ex - sx);
                const sh = Math.max(1, ey - sy);

                const cropCanvas = document.createElement("canvas");
                cropCanvas.width = sw;
                cropCanvas.height = sh;
                const cropCtx = cropCanvas.getContext("2d");
                cropCtx.drawImage(canvas, sx, sy, sw, sh, 0, 0, sw, sh);
                sourceCanvas = cropCanvas;
            }

            const blob = await new Promise(resolve => sourceCanvas.toBlob(resolve, "image/jpeg", 0.9));
            const fd = new FormData();
            fd.append("file", blob, "frame.jpg");
            fd.append("device_type", deviceType);

            const resp = await fetch("/detect", { method: "POST", body: fd });
            const data = await resp.json();
            if (!resp.ok) {
                resultBox.innerHTML = `<div class="small">Error: ${data.detail || "Detection failed"}</div>`;
                return;
            }

            const isBad = data.result === "NON-COMPLIANT";
            resultBox.innerHTML = `
                <div class="badge ${isBad ? "bad" : ""}">${data.result}</div>
                <div style="margin-top:8px">Score: ${data.score.toFixed(3)}</div>
                <div>Confidence: ${Math.round(data.confidence * 100)}%</div>
                <div class="small">Profile: ${data.device_type} | Threshold: ${data.threshold.toFixed(2)}</div>
                <div class="small" style="margin-top:8px">Logged: ${data.logged_at.date} ${data.logged_at.time}</div>
                <div class="small">Telegram: ${data.telegram_alert_sent ? "✓ Sent" : "- Not sent"}</div>
            `;
            
            // Audio feedback
            const speechText = isBad ? "Please follow proper dress code" : "Thank you, you may enter";
            playAudio(speechText);
            
            lastCaptureTime = Date.now();
        }

        function playAudio(text) {
            if ('speechSynthesis' in window) {
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.rate = 1.0;
                utterance.pitch = 1.0;
                utterance.volume = 1.0;
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(utterance);
            }
        }

        async function monitorAndAutoCapture() {
            if (!isRunning) return;

            try {
                const personFound = await detectPersonInFrame();
                const now = Date.now();

                if (personFound) {
                    if (!personDetectedSince) {
                        personDetectedSince = now;
                        statusBar.className = "status-bar person-found";
                        statusBar.innerHTML = `<div class="small">Person detected on ${deviceType}, auto-capture in 5s...</div>`;
                    }
                    const secondsSincePerson = (now - personDetectedSince) / 1000;
                        statusBar.innerHTML = `<div class="small person-found">Person detected (${secondsSincePerson.toFixed(1)}s). Auto-capturing every 5s on ${deviceType} profile...</div>`;

                    if (!lastCaptureTime || (now - lastCaptureTime) >= 5000) {
                        await detectOnce();
                    }
                } else {
                    personDetectedSince = null;
                    statusBar.className = "status-bar";
                    statusBar.innerHTML = `<div class="small">No person detected. Waiting...</div>`;
                }
            } catch (err) {
                console.error("Monitor error:", err);
            }

            setTimeout(monitorAndAutoCapture, 500);
        }

        async function startLive() {
            try {
                await startCamera();
                isRunning = true;
                monitorAndAutoCapture();
            } catch (err) {
                resultBox.innerHTML = `<div class="small">Camera error: ${err.message}</div>`;
                isRunning = false;
            }
        }

        function stopLive() {
            isRunning = false;
            personDetectedSince = null;
            lastCaptureTime = null;
            stopCamera();
            statusBar.className = "status-bar";
            statusBar.innerHTML = `<div class="small">Stopped.</div>`;
            resultBox.innerHTML = `<div class="small">Stopped.</div>`;
        }

        startBtn.addEventListener("click", startLive);
        stopBtn.addEventListener("click", stopLive);

        loadModel().catch(err => {
            statusBar.innerHTML = `<div class="small" style="color:red;">Model load failed: ${err.message}</div>`;
        });
    </script>
</body>
</html>
"""


@app.get("/status")
def status() -> Dict[str, Any]:
    return {
        "message": "Dress Code Detector API is running",
        "endpoints": {
            "home": "/",
            "status": "/status",
            "health": "/health",
            "detect": "/detect",
            "reports_today": "/reports/today",
            "reports_summary": "/reports/summary",
            "telegram_config": "/telegram-config",
            "docs": "/docs",
        },
    }


@app.get("/telegram-config")
def telegram_config() -> Dict[str, Any]:
    return {
        "telegram_bot_token_set": bool(TELEGRAM_BOT_TOKEN),
        "telegram_chat_id_set": bool(TELEGRAM_CHAT_ID),
        "ready": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
        "note": "If ready=false, set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars and restart service",
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok" if model is not None else "error",
        "model_loaded": model is not None,
        "model_path": resolve_model_path() if model is not None else MODEL_PATH,
        "input_size": [img_w, img_h],
    }


@app.get("/reports/today")
def reports_today(limit: int = 200) -> Dict[str, Any]:
    today = datetime.now().strftime("%Y-%m-%d")
    detail_file = os.path.join(REPORTS_DIR, f"detections_{today}.csv")
    rows = read_csv_rows(detail_file, limit=limit)
    return {
        "date": today,
        "count": len(rows),
        "rows": rows,
    }


@app.get("/reports/summary")
def reports_summary(limit: int = 365) -> Dict[str, Any]:
    rows = read_csv_rows(SUMMARY_FILE, limit=limit)
    return {
        "count": len(rows),
        "rows": rows,
    }


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    device_type: str = Form("unknown"),
) -> Dict[str, Any]:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image file")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Unable to decode image")

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        prediction = classify_image(image, device_type=device_type)
        logged_at = log_detection(
            result=prediction["result"],
            score=prediction["score"],
            confidence=prediction["confidence"],
        )
        telegram_sent = maybe_send_telegram_alert(
            image_bgr=image,
            result=prediction["result"],
            confidence=prediction["confidence"],
        )
        prediction["logged_at"] = {"date": logged_at["date"], "time": logged_at["time"]}
        prediction["telegram_alert_sent"] = telegram_sent
        return prediction
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")