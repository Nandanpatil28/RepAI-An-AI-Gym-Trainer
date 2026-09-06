"""Reusable, stateful pose-processing pipeline for the web application."""

from __future__ import annotations

import threading

import cv2
import mediapipe as mp

from exercise_classifier import ExerciseClassifier
from types_of_exercise import TypeOfExercise


ALL_EXERCISES = ("push-up", "pull-up", "sit-up", "squat", "walk")
DISPLAY_NAMES = {
    "push-up": "Push-Up", "pull-up": "Pull-Up", "sit-up": "Sit-Up",
    "squat": "Squat", "walk": "Walking",
}


class WorkoutEngine:
    """Owns MediaPipe and workout state for one browser workout session."""

    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        # At 10 submitted frames/sec, 15 frames is about 1.5 seconds.
        self.classifier = ExerciseClassifier(buffer_size=25, switch_threshold=15,
                                             min_confidence=0.35)
        self.counters = {exercise: 0 for exercise in ALL_EXERCISES}
        self.statuses = {exercise: True for exercise in ALL_EXERCISES}
        self.current_exercise = "squat"
        self.auto_mode = True
        self.confidence = 0.0
        self.transitioning = False
        self.lock = threading.Lock()

    def process(self, frame):
        """Process a BGR browser frame and return only lightweight HUD data."""
        data, _ = self._process(frame, draw_landmarks=False)
        return data

    def process_with_overlay(self, frame):
        """Process a BGR frame and return HUD data plus a landmark-annotated frame."""
        return self._process(frame, draw_landmarks=True)

    def _process(self, frame, draw_landmarks):
        with self.lock:
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = self.pose.process(rgb)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                if self.auto_mode:
                    detected, confidence, transitioning = self.classifier.update(landmarks)
                    self.confidence = confidence
                    self.transitioning = transitioning
                    if detected and detected != self.current_exercise:
                        self.current_exercise = detected

                count, status = TypeOfExercise(landmarks).calculate_exercise(
                    self.current_exercise,
                    self.counters[self.current_exercise],
                    self.statuses[self.current_exercise],
                )
                self.counters[self.current_exercise] = count
                self.statuses[self.current_exercise] = status
            else:
                self.transitioning = False

            if draw_landmarks and result.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                    mp.solutions.drawing_utils.DrawingSpec(color=(201, 255, 68), thickness=2, circle_radius=2),
                )
            return self.snapshot(bool(result.pose_landmarks)), frame

    def control(self, action, exercise=None):
        """Apply a user keyboard/button command to this workout only."""
        with self.lock:
            if action == "manual" and exercise in ALL_EXERCISES:
                self.current_exercise = exercise
                self.auto_mode = False
                self.classifier.set_exercise(exercise)
                self.confidence = 1.0
                self.transitioning = False
            elif action == "auto":
                self.auto_mode = True
                self.classifier.reset()
                self.transitioning = False
            elif action == "reset":
                self.counters[self.current_exercise] = 0
                self.statuses[self.current_exercise] = True
            elif action == "clear":
                self.counters = {item: 0 for item in ALL_EXERCISES}
                self.statuses = {item: True for item in ALL_EXERCISES}
            return self.snapshot()

    def snapshot(self, pose_detected=True):
        active_count = self.counters[self.current_exercise]
        status = "READY / UP" if self.statuses[self.current_exercise] else "IN MOTION / DOWN"
        if not pose_detected:
            status = "NO POSE DETECTED"
        return {
            "mode": DISPLAY_NAMES[self.current_exercise],
            "mode_key": self.current_exercise,
            "mode_type": "AUTO" if self.auto_mode else "MANUAL",
            "reps": active_count,
            "status": status,
            "confidence": round(self.confidence * 100),
            "transitioning": self.transitioning,
            "counters": {
                "pushup": self.counters["push-up"],
                "pullup": self.counters["pull-up"],
                "situp": self.counters["sit-up"],
                "squat": self.counters["squat"],
                "walk": self.counters["walk"],
            },
        }

    def close(self):
        self.pose.close()
