"""Read credentials before camera, network, or output side effects."""
import os

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
if not ROBOFLOW_API_KEY:
    raise RuntimeError("Set ROBOFLOW_API_KEY in your environment; never commit it.")
