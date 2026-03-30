#  Dress Code Detector using Computer Vision

##  Project Description

The Dress Code Detector is a smart monitoring and access control system built using Raspberry Pi, OpenCV, and a MobileNetV2-based classifier trained on 2.5k images in Google Colab. It captures live video using a webcam, detects whether the person follows proper dress code, and performs automated actions such as:

- Voice feedback
- Telegram alert notification
- LED indication
- CSV report generation
- Automatic door opening (optional)

##  Objectives

- Detect dress code compliance in real-time
- Automatically identify dress code violations
- Send Telegram notification for violations
- Provide instant voice feedback
- Maintain daily CSV report logs
- Control hardware components like LEDs and servo motor
  
##  Dress Code Rules
Girls:
- Long Kurthas

Boys:
- Formal Shirt
- Formal Pant

##  Features

### 1. Real-Time Detection
- Captures live video using USB webcam
- Uses a MobileNetV2 model trained on 2.5k labeled images (Google Colab) for detection
- Processes frames using OpenCV
  
### 2. Telegram Alert System

When dress code violation is detected:

- Sends automatic message to Telegram channel
- Includes timestamp information

### 3. Daily Report Generation

The system creates and updates a CSV file automatically.

Report includes:
- Date
- Time
- Compliance Status

### 4. Voice Feedback System

Provides instant audio feedback using Bluetooth speaker.

Compliance:
```
Thank you, you may enter
```

Non-compliance:
```
Please follow proper dress code
```

##  Technologies Used

### Software

- Python
- OpenCV
- MobileNetV2 (TensorFlow/Keras)
- NumPy
- Pandas
- pyttsx3 (Text to Speech)
- Requests (Telegram API)

### Hardware

- Raspberry Pi 4
- USB Webcam
- Micro SD Card
- Power Supply
- Monitor (Laptop used)
- Servo Motor
- Bluetooth Speaker
  
##  System Architecture

```
Camera Input
     ↓
OpenCV Capture
     ↓
MobileNetV2 Inference
     ↓
Decision Logic
     ↓
 ┌──────────────┬──────────────┬──────────────┬──────────────┐
 ↓              ↓              ↓              ↓
Voice Feedback  LED Control   Telegram Alert CSV Report
                                      ↓
                                 Servo Motor (Optional)
```

##  Applications

- College entry monitoring
- Lab entry control
- Corporate dress monitoring
- Hostel entry monitoring
- Secure access systems

##  Render Deployment (Shareable Camera Link)

After deployment, users can open your Render URL directly on phone or PC and run live detection with their own camera.

- Home page (camera UI): /
- Health check: /health
- API docs: /docs
- Live detection API: /detect
- Today logs: /reports/today
- Summary logs: /reports/summary

### Telegram setup on Render

Set these environment variables in Render:

- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID
- TELEGRAM_ALERT_COOLDOWN_SEC (optional, default 30)

### Camera tuning for phone and laptop

This project now supports separate tuning for phone and laptop camera feeds.

- DRESS_THRESHOLD: global default threshold (fallback)
- DRESS_THRESHOLD_PHONE: threshold used when source is phone camera
- DRESS_THRESHOLD_LAPTOP: threshold used when source is laptop camera

For local OpenCV app (`realtime_detector.py`):

- CAMERA_PROFILE=laptop (default) or CAMERA_PROFILE=phone

Suggested starting values:

- DRESS_THRESHOLD=0.50
- DRESS_THRESHOLD_PHONE=0.58
- DRESS_THRESHOLD_LAPTOP=0.52

### Important note about CSV persistence on Render

CSV files written inside the service filesystem may be lost on restart/redeploy unless you use a persistent disk or external database/storage.



