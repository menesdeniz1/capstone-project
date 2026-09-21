# M&M detection and sorting-integration experiments

University team capstone exploring object/color detection with camera input and serial/Bluetooth communication. The repository preserves experimental Python scripts, model artifacts and an Arduino receiver. It is not a validated industrial sorting system.

## Code map

- `realTimeArduino.py`: local Ultralytics YOLO detector with capture/inference/display threads, two bounded queues and serial messages. Defaults assume Windows DirectShow camera 0 and serial port COM9.
- `receiver.ino`: receives newline-delimited Bluetooth messages and displays them on an I2C LCD. This file does **not** implement servo sorting control.
- `mmdetector.py`, `realtimedetector.py`, `snapsavedata.py`: separate Roboflow workflow experiments; provider workspace settings remain project-specific.
- `train.py`, `test.py`, `single_detection.py`, `bestt.py`, `working_realtime_yolo.py`: historical training/detection variants. Some contain machine-specific paths and are not interchangeable entry points.

## Local detector setup

In a Python virtual environment, install `ultralytics`, `opencv-python`, `numpy` and `pyserial`. Review the script and set `CAPSTONE_MODEL_PATH` to a trusted compatible checkpoint if overriding the included default; set `CAPSTONE_SERIAL_PORT` for your device. Run `python realTimeArduino.py` only when you intend to open the camera and communicate with attached hardware.

Roboflow variants additionally require their own inference dependencies and `ROBOFLOW_API_KEY` in the process environment. `.env.example` is documentation only and is not automatically loaded. See [configuration notes](PUBLICATION_NOTES.md).

## Limitations and attribution

The configured 30 FPS is a target, **not a measured achievement**. Neither hardware operation, sorting accuracy, throughput nor production deployment was validated during maintenance. Python syntax checks are not hardware tests. Do not load untrusted model files. Dataset and pretrained-model terms remain separate from authorship of the application code; no blanket license is granted for third-party artifacts.

Team authorship is retained in Git history. Embedded credentials were removed from reachable history, but this does not revoke the provider key or remove copies elsewhere. Previously exposed Roboflow keys were left unchanged at the owner's explicit request; that exposure is not represented as resolved.
