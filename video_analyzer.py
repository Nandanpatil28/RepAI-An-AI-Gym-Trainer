"""Sequential uploaded-video analysis using the existing WorkoutEngine."""

from __future__ import annotations

import cv2

from exercise_engine import WorkoutEngine


class VideoAnalyzer:
    """Streams a file from OpenCV; it never loads the full video into memory."""

    def __init__(self, path):
        self.path = str(path)
        self.capture = cv2.VideoCapture(self.path)
        if not self.capture.isOpened():
            raise ValueError("OpenCV could not open this video.")
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
        if self.frame_count <= 0:
            self.capture.release()
            raise ValueError("The video does not contain readable frames.")
        self.duration_seconds = round(self.frame_count / self.fps, 2)
        self.frame_index = 0
        self.engine = WorkoutEngine()
        self.complete = False
        self.saved = False
        self.person_frames = 0
        self.detected_exercises = set()
        self.sampled_frames = 0
        self.playback_seconds = 0.0
        self.ai_fps = 10

    def process_browser_frame(self, frame, timestamp_seconds):
        """Process a sampled frame without controlling browser video playback.

        The browser owns native playback and submits at most ``ai_fps`` frames per
        second. This deliberately decouples MediaPipe latency from the video clock.
        """
        data = self.engine.process(frame)
        self.sampled_frames += 1
        self.playback_seconds = max(0.0, min(float(timestamp_seconds), self.duration_seconds))
        self.frame_index = min(self.frame_count, int(round(self.playback_seconds * self.fps)))
        if data["status"] != "NO POSE DETECTED":
            self.person_frames += 1
            self.detected_exercises.add(data["mode"])
        return self.payload(data)

    def payload(self, data=None, image=None):
        data = data or self.engine.snapshot(self.person_frames > 0)
        data.update({
            "frame": self.frame_index,
            "frame_count": self.frame_count,
            "progress": round((self.frame_index / self.frame_count) * 100) if self.frame_count else 0,
            "duration_seconds": self.duration_seconds,
            "source_fps": round(self.fps, 3),
            "ai_fps": self.ai_fps,
            "sampled_frames": self.sampled_frames,
            "complete": self.complete,
            "person_detected": self.person_frames > 0,
            "detected_exercises": sorted(self.detected_exercises),
            "image": image,
        })
        return data

    def close(self):
        self.capture.release()
        self.engine.close()
