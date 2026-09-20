# Maintenance and setup notes

University team project exploring M&M detection and color sorting. The repository contains experimental scripts; do not treat a target FPS value as a measured performance result.

## Credential configuration

The Roboflow scripts read `ROBOFLOW_API_KEY` from the process environment before opening a camera or starting network activity. Supply a new key you are authorized to use. The `.env.example` file documents the variable only; the scripts do not automatically load it.

In PowerShell, set the environment variable locally, then run the selected script. Do not paste real credentials into documentation, source files, screenshots, or logs.

- `mmdetector.py`: local Roboflow workflow experiment.
- `realtimedetector.py`: live camera inference client.
- `snapsavedata.py`: capture and save inference output.

Workspace/workflow identifiers remain project-specific. Camera access, service availability, group dataset rights, and end-to-end operation still need verification. No live inference or hardware test was performed during this maintenance pass.

## History and publication

Embedded API credentials were removed from reachable Git history. Commit authors and work sequence are retained, but commit IDs changed. Do not push from old clones. Credential owners must still revoke previously exposed keys; old external copies and cached commits are not erased by this operation. Keep this repository private until those checks are resolved.
