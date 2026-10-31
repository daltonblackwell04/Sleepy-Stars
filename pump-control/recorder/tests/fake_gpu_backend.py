"""Scripted stand-in for the YOLO GPU backend, loaded inside the StarSeg GPU child process.

model_path points at a JSON file: {"names": {...}, "task": "segment", "actions": [...]}.
Each predict call consumes the next action, counted in <model_path>.count so the script
continues across child restarts. Actions: "ok", "cuda" (fatal CUDA error), "value"
(ordinary error), "hang" (sleep far past the timeout), "die" (child exits without replying).
"""
import json
import os
import time
from pathlib import Path


class Backend:
    device = "cuda-fake"

    def __init__(self, model_path: str):
        self.path = Path(model_path)
        spec = json.loads(self.path.read_text())
        self.names = spec.get("names", {"0": "sunflower_star"})
        self.task = spec.get("task", "segment")
        self.actions = spec.get("actions", [])

    def _next(self) -> str:
        counter = self.path.with_suffix(".count")
        n = int(counter.read_text()) if counter.exists() else 0
        counter.write_text(str(n + 1))
        return self.actions[n] if n < len(self.actions) else "ok"

    def predict(self, images, conf, imgsz):
        action = self._next()
        if action == "cuda":
            raise RuntimeError("CUDA error: the launch timed out and was terminated")
        if action == "value":
            raise ValueError("bad input frame")
        if action == "hang":
            time.sleep(600)
        if action == "die":
            os._exit(3)
        return [[{"class_id": 0, "class_name": "sunflower_star", "confidence": 0.9,
                  "box_xyxy": [0.0, 0.0, 1.0, 1.0], "segment_xy": [], "pid": os.getpid(),
                  "shape": list(getattr(image, "shape", ()))}]
                for image in images]
