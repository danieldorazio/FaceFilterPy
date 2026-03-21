# FaceFilterPy 🕶️🎩

Real-time face filter in Python using MediaPipe and OpenCV.
Detects your face and hands via webcam and lets you apply accessories
(sunglasses, hats) through an interactive sidebar.

## Demo
> Pick your accessories with a click and wear them in real time 😄

## Tech Stack
- Python 3.11
- MediaPipe 0.10.9 — face mesh + hand tracking
- OpenCV 4.13

## Features
- 🕶️ Glasses overlay — aligned and rotated to follow your eyes
- 🎩 Hat overlay — aligned to head width and position
- 🖱️ Interactive sidebar — click to activate/deactivate accessories
- 🔄 Toggle between multiple glasses and hats with a single click
- 📍 468 face mesh landmarks visible in debug mode
- 🖐️ 21 hand landmarks per hand (up to 2 hands) visible in debug mode
- 🖥️ Fullscreen support
- 🔁 LERP smoothing on all accessory positions to eliminate flickering
- 🐛 Debug mode toggle with D key

## How it works
1. MediaPipe Face Mesh detects 468 landmarks on your face every frame
2. Glasses are placed using landmarks 33 and 263 (eye corners)
3. Hat is placed using landmarks 234, 454 (head width) and 10 (forehead)
4. LERP smoothing (factor 0.7) is applied to eliminate flickering
5. Sidebar on the right shows all available accessories grouped by category
6. Click an accessory to activate it — click again to deactivate

## Landmark reference
```
33:  left eye outer corner
263: right eye outer corner
234: left cheekbone (head width)
454: right cheekbone (head width)
10:  forehead top
```

## Requirements
- Python 3.11
- Webcam

## Installation
```bash
py -3.11 -m pip install opencv-python mediapipe==0.10.9
```

## Run
```bash
py -3.11 main.py
```

## Controls
| Key | Action |
|-----|--------|
| ESC | Exit |
| D   | Toggle debug landmarks |
| Mouse click on sidebar | Activate/deactivate accessory |

## Folder structure
```
FaceFilterPy/
  img/
    glasses/
      glasses1.png
      glasses2.png
      ...
    hats/
      hat1.png
      hat2.png
      ...
  main.py
  README.md
```

## Notes
- Compatible with MediaPipe 0.10.9 (newer versions changed the API)
- Images must be PNG with transparent background
- Add more accessories by dropping PNG files in the respective folders