"""Reusable, stateful pose-processing pipeline for the web application."""

from __future__ import annotations

import threading
import time

import cv2
import mediapipe as mp

from exercise_classifier import ExerciseClassifier
from utils import detection_body_part, calculate_angle


ALL_EXERCISES = ("push-up", "pull-up", "sit-up", "squat", "walk")
DISPLAY_NAMES = {
    "push-up": "Push-Up", "pull-up": "Pull-Up", "sit-up": "Sit-Up",
    "squat": "Squat", "walk": "Walking",
}

REQUIRED_LANDMARKS = {
    "push-up": ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
                "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP"),
    "pull-up": ("NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW",
                "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP"),
    "sit-up": ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
               "LEFT_KNEE", "RIGHT_KNEE"),
    "squat": ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
              "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"),
    "walk": ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
             "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"),
}

POSE_VISIBILITY_THRESHOLDS = {
    "push-up": 0.20,
    "pull-up": 0.55,
    "sit-up": 0.25,
    "squat": 0.45,
    "walk": 0.58,
}

PHASE_STABLE_FRAMES = {
    "push-up": 2,
    "pull-up": 2,
    "sit-up": 2,
    "squat": 2,
    "walk": 1,
}


def average_angle(left, middle, right):
    return calculate_angle(left, middle, right)


def generate_feedback(exercise, landmarks, counters, confidence, is_transitioning):
    """Generate exercise-specific real-time feedback from pose landmarks.

    Returns a list of feedback strings (may be empty when no correction is
    needed or when there is insufficient confidence / data).
    """
    feedback = []
    if landmarks is None or exercise is None:
        return feedback

    try:
        # Only provide feedback when the classifier is reasonably confident
        if confidence < 0.35 or is_transitioning:
            return feedback

        if exercise == "push-up":
            l_shoulder = detection_body_part(landmarks, "LEFT_SHOULDER")
            r_shoulder = detection_body_part(landmarks, "RIGHT_SHOULDER")
            l_hip = detection_body_part(landmarks, "LEFT_HIP")
            r_hip = detection_body_part(landmarks, "RIGHT_HIP")
            wrist_y = (detection_body_part(landmarks, "LEFT_WRIST")[1] +
                       detection_body_part(landmarks, "RIGHT_WRIST")[1]) / 2
            shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
            hip_y = (l_hip[1] + r_hip[1]) / 2
            torso_delta = abs(shoulder_y - hip_y)
            if torso_delta > 0.15:
                feedback.append("Keep your body straight from head to heels.")
            if wrist_y < shoulder_y - 0.10:
                feedback.append("Lower your chest closer to the ground.")
            if wrist_y > hip_y + 0.10:
                feedback.append("Don't let your hips sag — engage your core.")
            if not feedback:
                feedback.append("Good form — keep going!")

        elif exercise == "pull-up":
            l_wrist = detection_body_part(landmarks, "LEFT_WRIST")
            r_wrist = detection_body_part(landmarks, "RIGHT_WRIST")
            l_shoulder = detection_body_part(landmarks, "LEFT_SHOULDER")
            shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
            wrist_y = (l_wrist[1] + r_wrist[1]) / 2
            if wrist_y > shoulder_y:
                feedback.append("Grip the bar above you — wrists should be over the bar.")
            shoulder_angle = calculate_angle(
                detection_body_part(landmarks, "LEFT_ELBOW"),
                l_shoulder,
                detection_body_part(landmarks, "LEFT_HIP"),
            )
            if shoulder_angle < 30:
                feedback.append("Pull your chin over the bar — you're almost there!")
            if not feedback:
                feedback.append("Strong pull — control the descent!")

        elif exercise == "squat":
            l_shoulder = detection_body_part(landmarks, "LEFT_SHOULDER")
            r_shoulder = detection_body_part(landmarks, "RIGHT_SHOULDER")
            l_hip = detection_body_part(landmarks, "LEFT_HIP")
            r_hip = detection_body_part(landmarks, "RIGHT_HIP")
            l_ankle = detection_body_part(landmarks, "LEFT_ANKLE")
            r_ankle = detection_body_part(landmarks, "RIGHT_ANKLE")
            knee_x = (detection_body_part(landmarks, "LEFT_KNEE")[0] +
                      detection_body_part(landmarks, "RIGHT_KNEE")[0]) / 2
            l_ankle_x = l_ankle[0]
            r_ankle_x = r_ankle[0]
            # Knee alignment check
            if knee_x < l_ankle_x - 0.05:
                feedback.append("Push your knees over your toes.")
            elif knee_x > r_ankle_x + 0.05:
                feedback.append("Keep knees aligned over ankles.")
            hip_y = (l_hip[1] + r_hip[1]) / 2
            shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
            if hip_y > shoulder_y + 0.15:
                feedback.append("Keep your chest up — don't hunch over.")
            if not feedback:
                feedback.append("Good depth — stand up tall!")

        elif exercise == "sit-up":
            l_shoulder = detection_body_part(landmarks, "LEFT_SHOULDER")
            r_shoulder = detection_body_part(landmarks, "RIGHT_SHOULDER")
            l_hip = detection_body_part(landmarks, "LEFT_HIP")
            r_hip = detection_body_part(landmarks, "RIGHT_HIP")
            shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
            hip_y = (l_hip[1] + r_hip[1]) / 2
            if shoulder_y > hip_y:
                feedback.append("Lift your shoulders off the ground.")
            elif shoulder_y < hip_y - 0.25:
                feedback.append("Lower back down slowly to start position.")
            l_elbow = detection_body_part(landmarks, "LEFT_ELBOW")
            l_shoulder = detection_body_part(landmarks, "LEFT_SHOULDER")
            if l_elbow[0] < l_shoulder[0] - 0.3:
                feedback.append("Place your hands by your ears, not behind your head.")
            if not feedback:
                feedback.append("Nice lift — engage your core!")

        elif exercise == "walk":
            feedback.append("Keep your head up and swing your arms naturally.")

    except Exception:
        pass

    return feedback[:2]  # Max 2 concurrent feedback messages


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
        self.rep_states = {exercise: self._new_rep_state() for exercise in ALL_EXERCISES}
        self.current_exercise = "squat"
        self.auto_mode = True
        self.confidence = 0.0
        self.transitioning = False
        self.feedback = []
        self.pose_quality = "waiting"
        self.pose_message = "Waiting for person."
        self.start_time = None
        self.elapsed_before_pause = 0.0
        self.paused_at = None
        self.last_pose_time = None
        self.lock = threading.Lock()

    def _new_rep_state(self):
        return {
            "phase": "wait_start",
            "stable": 0,
            # Walking is tracked separately from repetitions. A step is only
            # accepted when the next raised foot alternates.
            "walk_last_leg": None,
            "walk_candidate_leg": None,
            "walk_candidate_frames": 0,
            "walk_cooldown": 0,
            "walk_grounded": True,
            "walk_armed": False,
            "walk_previous_ankles": None,
        }

    def _reset_rep_state(self, exercise=None):
        if exercise:
            self.rep_states[exercise] = self._new_rep_state()
            self.statuses[exercise] = True
            return
        self.rep_states = {item: self._new_rep_state() for item in ALL_EXERCISES}
        self.statuses = {item: True for item in ALL_EXERCISES}

    def process(self, frame, draw_landmarks=False):
        """Process a BGR browser frame and return lightweight HUD data.

        When ``draw_landmarks`` is True, the returned tuple also includes
        the annotated BGR frame so the caller can stream it back to the
        browser to mirror the local ``main.py`` overlay.
        """
        data, annotated = self._process(frame, draw_landmarks=draw_landmarks)
        return (data, annotated) if draw_landmarks else data

    def process_with_overlay(self, frame):
        """Process a BGR frame and return HUD data plus a landmark-annotated frame."""
        return self._process(frame, draw_landmarks=True)

    def _landmark(self, landmarks, name):
        return detection_body_part(landmarks, name)

    def _pose_quality_for(self, landmarks, exercise):
        required = REQUIRED_LANDMARKS[exercise]
        min_visibility = POSE_VISIBILITY_THRESHOLDS.get(exercise, 0.58)
        visible = []
        for name in required:
            point = self._landmark(landmarks, name)
            x, y, visibility = point
            if visibility < min_visibility:
                return False, "unreliable", "Please adjust your position so your full body is visible."
            if not (-0.08 <= x <= 1.08 and -0.08 <= y <= 1.08):
                return False, "unreliable", "Please adjust your position so your full body is visible."
            visible.append(point)
        min_x = min(point[0] for point in visible)
        max_x = max(point[0] for point in visible)
        min_y = min(point[1] for point in visible)
        max_y = max(point[1] for point in visible)
        if max_x - min_x < 0.08 or max_y - min_y < 0.12:
            return False, "unreliable", "Please adjust your position so your full body is visible."
        return True, "tracking", "Pose tracking."

    def _angles(self, landmarks):
        l_shoulder = self._landmark(landmarks, "LEFT_SHOULDER")
        r_shoulder = self._landmark(landmarks, "RIGHT_SHOULDER")
        l_elbow = self._landmark(landmarks, "LEFT_ELBOW")
        r_elbow = self._landmark(landmarks, "RIGHT_ELBOW")
        l_wrist = self._landmark(landmarks, "LEFT_WRIST")
        r_wrist = self._landmark(landmarks, "RIGHT_WRIST")
        l_hip = self._landmark(landmarks, "LEFT_HIP")
        r_hip = self._landmark(landmarks, "RIGHT_HIP")
        l_knee = self._landmark(landmarks, "LEFT_KNEE")
        r_knee = self._landmark(landmarks, "RIGHT_KNEE")
        l_ankle = self._landmark(landmarks, "LEFT_ANKLE")
        r_ankle = self._landmark(landmarks, "RIGHT_ANKLE")

        shoulder_mid = [(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2]
        hip_mid = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]
        knee_mid = [(l_knee[0] + r_knee[0]) / 2, (l_knee[1] + r_knee[1]) / 2]
        elbow_mid = [(l_elbow[0] + r_elbow[0]) / 2, (l_elbow[1] + r_elbow[1]) / 2]

        return {
            "arm": (average_angle(l_shoulder, l_elbow, l_wrist) +
                    average_angle(r_shoulder, r_elbow, r_wrist)) / 2,
            "leg": (average_angle(l_hip, l_knee, l_ankle) +
                    average_angle(r_hip, r_knee, r_ankle)) / 2,
            "abdomen": calculate_angle(shoulder_mid, hip_mid, knee_mid),
            "nose_y": self._landmark(landmarks, "NOSE")[1],
            "elbow_y": elbow_mid[1],
            "shoulder_y": shoulder_mid[1],
            "hip_y": hip_mid[1],
            "knee_y": knee_mid[1],
            "left_knee_x": l_knee[0],
            "right_knee_x": r_knee[0],
            "left_ankle_y": l_ankle[1],
            "right_ankle_y": r_ankle[1],
            "torso_y_delta": abs(shoulder_mid[1] - hip_mid[1]),
        }

    def _update_phase_counter(self, exercise, start_position, action_position):
        state = self.rep_states[exercise]
        stable_frames = PHASE_STABLE_FRAMES.get(exercise, 3)
        counted = False
        if state["phase"] == "wait_start":
            if start_position:
                state["stable"] += 1
                if state["stable"] >= stable_frames:
                    state["phase"] = "ready"
                    state["stable"] = 0
                    self.statuses[exercise] = True
            else:
                state["stable"] = 0
        elif state["phase"] == "ready":
            if action_position:
                state["stable"] += 1
                if state["stable"] >= stable_frames:
                    state["phase"] = "action"
                    state["stable"] = 0
                    self.statuses[exercise] = False
            else:
                state["stable"] = 0
        elif state["phase"] == "action":
            if start_position:
                state["stable"] += 1
                if state["stable"] >= stable_frames:
                    self.counters[exercise] += 1
                    counted = True
                    state["phase"] = "ready"
                    state["stable"] = 0
                    self.statuses[exercise] = True
            else:
                state["stable"] = 0
        return counted

    def _count_complete_movement(self, exercise, landmarks):
        metrics = self._angles(landmarks)
        if exercise == "push-up":
            horizontal = metrics["torso_y_delta"] < 0.24
            return self._update_phase_counter(exercise, metrics["arm"] > 155 and horizontal,
                                              metrics["arm"] < 95 and horizontal)
        if exercise == "squat":
            upright = metrics["shoulder_y"] < metrics["hip_y"] < metrics["knee_y"]
            return self._update_phase_counter(exercise, metrics["leg"] > 135 and upright,
                                              metrics["leg"] < 130 and upright)
        if exercise == "sit-up":
            return self._update_phase_counter(exercise, metrics["abdomen"] > 105,
                                              metrics["abdomen"] < 75)
        if exercise == "pull-up":
            hanging = metrics["arm"] > 135 and metrics["nose_y"] > metrics["elbow_y"] + 0.03
            top = metrics["arm"] < 120 and metrics["nose_y"] < metrics["elbow_y"]
            return self._update_phase_counter(exercise, hanging, top)
        if exercise == "walk":
            return self._count_walk_step(exercise, metrics)
        return False

    def _count_walk_step(self, exercise, metrics):
        """Count steps from alternating, lifted ankle landmarks.

        Image y coordinates increase downward: the foot with the lower y value
        is raised. Two stable frames, alternating legs, and a cooldown reject
        landmark jitter and prevent a held pose from counting repeatedly.
        """
        state = self.rep_states[exercise]
        state["walk_cooldown"] = max(0, state["walk_cooldown"] - 1)
        upright = metrics["torso_y_delta"] > 0.13
        if not upright or metrics["leg"] < 115:
            return False

        left_y, right_y = metrics["left_ankle_y"], metrics["right_ankle_y"]
        separation = abs(left_y - right_y)
        previous = state["walk_previous_ankles"]
        state["walk_previous_ankles"] = (left_y, right_y)
        if separation < 0.025:
            state["walk_grounded"] = True
            state["walk_armed"] = True
            state["walk_candidate_leg"] = None
            state["walk_candidate_frames"] = 0
            self.statuses[exercise] = True
            return False
        if separation < 0.055 or previous is None:
            return False

        leg = "left" if left_y < right_y else "right"
        lift_velocity = (left_y - previous[0]) if leg == "left" else (right_y - previous[1])
        moving_into_step = state["walk_armed"] and (state["walk_grounded"] or lift_velocity < -0.004)
        if not moving_into_step:
            return False
        if state["walk_candidate_leg"] == leg:
            state["walk_candidate_frames"] += 1
        else:
            state["walk_candidate_leg"] = leg
            state["walk_candidate_frames"] = 1

        counted = (state["walk_candidate_frames"] >= 2
                   and state["walk_last_leg"] != leg
                   and state["walk_cooldown"] == 0)
        if counted:
            self.counters[exercise] += 1
            state["walk_last_leg"] = leg
            state["walk_cooldown"] = 3
            state["walk_grounded"] = False
            state["walk_candidate_frames"] = 0
        self.statuses[exercise] = not counted
        return counted

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
                        self._reset_rep_state(detected)
                else:
                    confidence = self.confidence
                    transitioning = self.transitioning

                pose_ok, quality, quality_message = self._pose_quality_for(landmarks, self.current_exercise)
                self.pose_quality = quality
                self.pose_message = quality_message
                if pose_ok and not self.transitioning:
                    self._count_complete_movement(self.current_exercise, landmarks)
                    self.feedback = generate_feedback(
                        self.current_exercise,
                        landmarks,
                        self.counters,
                        self.confidence,
                        self.transitioning,
                    )
                    if not self.feedback:
                        self.feedback = ["Tracking movement. Complete the full rep to count it."]
                else:
                    self.feedback = [quality_message]
            else:
                self.pose_quality = "missing"
                self.pose_message = "No person detected. Please position yourself in front of the camera."
                self.transitioning = False
                self.feedback = [self.pose_message]

            now = time.monotonic()
            if self.start_time is None:
                self.start_time = now
            self.last_pose_time = now

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
                self._reset_rep_state(exercise)
                self.pose_quality = "waiting"
                self.pose_message = "Waiting for person."
            elif action == "auto":
                self.auto_mode = True
                self.classifier.reset()
                self.transitioning = False
                self._reset_rep_state()
                self.pose_quality = "waiting"
                self.pose_message = "Waiting for person."
            elif action == "reset":
                self.counters[self.current_exercise] = 0
                self._reset_rep_state(self.current_exercise)
            elif action == "clear":
                self.counters = {item: 0 for item in ALL_EXERCISES}
                self._reset_rep_state()
                self.start_time = None
                self.elapsed_before_pause = 0.0
                self.paused_at = None
                self.last_pose_time = None
            elif action == "pause" and self.paused_at is None:
                now = time.monotonic()
                if self.start_time is not None:
                    self.elapsed_before_pause += now - self.start_time
                self.paused_at = now
                self.start_time = None
            elif action == "resume" and self.paused_at is not None:
                self.start_time = time.monotonic()
                self.paused_at = None
            return self.snapshot()

    def snapshot(self, pose_detected=True):
        active_count = self.counters[self.current_exercise]
        phase = self.rep_states[self.current_exercise]["phase"]
        status = "READY / START" if self.statuses[self.current_exercise] else "IN MOVEMENT"
        if phase == "wait_start":
            status = "FIND START POSITION"
        elif phase == "action":
            status = "RETURN TO START"
        if not pose_detected:
            status = "NO PERSON DETECTED"
        elif self.pose_quality == "unreliable":
            status = "POSE UNRELIABLE"

        # Calculate elapsed timer (seconds since workout started)
        elapsed = 0
        elapsed_sec = self.elapsed_before_pause
        if self.start_time is not None:
            elapsed_sec += time.monotonic() - self.start_time
        elapsed = int(elapsed_sec)

        return {
            "mode": DISPLAY_NAMES[self.current_exercise],
            "mode_key": self.current_exercise,
            "mode_type": "AUTO" if self.auto_mode else "MANUAL",
            "reps": active_count,
            "count_label": "Steps" if self.current_exercise == "walk" else "Repetitions",
            "count_unit": "steps" if self.current_exercise == "walk" else "reps",
            "status": status,
            "confidence": round(self.confidence * 100),
            "transitioning": self.transitioning,
            "timer": elapsed,
            "feedback": self.feedback,
            "pose_quality": self.pose_quality,
            "pose_message": self.pose_message,
            "rep_phase": phase,
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
