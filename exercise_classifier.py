"""
Exercise Classifier — Automatic exercise detection using MediaPipe pose landmarks.

Uses rule-based heuristic scoring for each exercise type, a rolling frame buffer
for temporal smoothing, and candidate stability tracking to prevent rapid switching.
"""

import numpy as np
from collections import deque
from body_part_angle import BodyPartAngle
from utils import detection_body_part, calculate_angle


class ExerciseClassifier:
    """Classifies the current exercise from pose landmarks with temporal stability."""

    EXERCISES = ["push-up", "pull-up", "sit-up", "squat", "walk"]

    def __init__(self, buffer_size=45, switch_threshold=30, min_confidence=0.35):
        """
        Args:
            buffer_size: Number of frames in the rolling score buffer (~1.5s at 30fps).
            switch_threshold: Consecutive frames a new candidate must lead before switching.
            min_confidence: Minimum averaged confidence required to consider a switch.
        """
        self.buffer_size = buffer_size
        self.switch_threshold = switch_threshold
        self.min_confidence = min_confidence

        # Rolling buffer of per-frame score dictionaries
        self.score_buffer = deque(maxlen=buffer_size)

        # Candidate tracking for temporal stability
        self.candidate_exercise = None
        self.candidate_frames = 0

        # Current detected exercise and confidence
        self.current_exercise = None
        self.current_confidence = 0.0

        # History of knee X positions for walking detection
        self.knee_history = deque(maxlen=30)

        # Transitioning state
        self.is_transitioning = False

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def update(self, landmarks):
        """
        Feed one frame of landmarks into the classifier.

        Returns:
            (exercise_name, confidence, is_transitioning)
            - exercise_name: str or None — the currently detected exercise.
            - confidence: float 0–1 — averaged confidence for the current exercise.
            - is_transitioning: bool — True while a new candidate is building up.
        """
        if landmarks is None:
            return self.current_exercise, self.current_confidence, False

        # Store knee positions for walking temporal analysis
        try:
            l_knee = detection_body_part(landmarks, "LEFT_KNEE")
            r_knee = detection_body_part(landmarks, "RIGHT_KNEE")
            self.knee_history.append((l_knee[0], r_knee[0]))
        except Exception:
            pass

        # Compute per-frame scores
        try:
            scores = self._compute_frame_scores(landmarks)
        except Exception:
            return self.current_exercise, self.current_confidence, False

        self.score_buffer.append(scores)

        # Need a minimum number of frames before making decisions
        if len(self.score_buffer) < 10:
            if self.current_exercise is None:
                # Pick the best from whatever we have so far
                best = max(scores, key=scores.get)
                if scores[best] >= self.min_confidence:
                    self.current_exercise = best
                    self.current_confidence = scores[best]
            return self.current_exercise, self.current_confidence, False

        # Average scores across the buffer
        avg_scores = {}
        for ex in self.EXERCISES:
            avg_scores[ex] = float(np.mean([s.get(ex, 0.0) for s in self.score_buffer]))

        # Find the best candidate
        best_exercise = max(avg_scores, key=avg_scores.get)
        best_confidence = avg_scores[best_exercise]

        # Update displayed confidence for the current exercise
        if self.current_exercise and self.current_exercise in avg_scores:
            self.current_confidence = avg_scores[self.current_exercise]

        # If no exercise set yet, accept the best immediately
        if self.current_exercise is None:
            if best_confidence >= self.min_confidence:
                self.current_exercise = best_exercise
                self.current_confidence = best_confidence
            return self.current_exercise, self.current_confidence, False

        # If the best is below minimum confidence, don't switch
        if best_confidence < self.min_confidence:
            self.candidate_exercise = None
            self.candidate_frames = 0
            self.is_transitioning = False
            return self.current_exercise, self.current_confidence, False

        # If the best is the same as current, stay put
        if best_exercise == self.current_exercise:
            self.candidate_exercise = None
            self.candidate_frames = 0
            self.is_transitioning = False
            self.current_confidence = best_confidence
            return self.current_exercise, self.current_confidence, False

        # A different exercise is leading — track candidate stability
        if best_exercise == self.candidate_exercise:
            self.candidate_frames += 1
        else:
            self.candidate_exercise = best_exercise
            self.candidate_frames = 1

        # Has the candidate been stable long enough?
        if self.candidate_frames >= self.switch_threshold:
            self.current_exercise = self.candidate_exercise
            self.current_confidence = best_confidence
            self.candidate_exercise = None
            self.candidate_frames = 0
            self.is_transitioning = False
            return self.current_exercise, self.current_confidence, False

        # Transition is building up but not complete
        self.is_transitioning = True
        return self.current_exercise, self.current_confidence, True

    def set_exercise(self, exercise):
        """Manually override the detected exercise (used when user presses 1–5)."""
        self.current_exercise = exercise
        self.current_confidence = 1.0
        self.candidate_exercise = None
        self.candidate_frames = 0
        self.is_transitioning = False
        self.score_buffer.clear()
        self.knee_history.clear()

    def reset(self):
        """Clear all state — used when toggling back to AUTO mode."""
        self.score_buffer.clear()
        self.knee_history.clear()
        self.candidate_exercise = None
        self.candidate_frames = 0
        self.is_transitioning = False
        # Keep current_exercise so there's no gap, but let classifier override it

    # -------------------------------------------------------------------------
    # Scoring heuristics
    # -------------------------------------------------------------------------

    def _compute_frame_scores(self, landmarks):
        """
        Compute a 0.0–1.0 confidence score for each exercise from a single frame.

        Uses body angles from BodyPartAngle and raw landmark positions to build
        a feature vector, then applies hand-crafted scoring rules.
        """
        body = BodyPartAngle(landmarks)

        # --- Extract key body part positions (normalised 0–1 coords) ---
        nose = detection_body_part(landmarks, "NOSE")
        l_shoulder = detection_body_part(landmarks, "LEFT_SHOULDER")
        r_shoulder = detection_body_part(landmarks, "RIGHT_SHOULDER")
        l_elbow = detection_body_part(landmarks, "LEFT_ELBOW")
        r_elbow = detection_body_part(landmarks, "RIGHT_ELBOW")
        l_wrist = detection_body_part(landmarks, "LEFT_WRIST")
        r_wrist = detection_body_part(landmarks, "RIGHT_WRIST")
        l_hip = detection_body_part(landmarks, "LEFT_HIP")
        r_hip = detection_body_part(landmarks, "RIGHT_HIP")
        l_knee = detection_body_part(landmarks, "LEFT_KNEE")
        r_knee = detection_body_part(landmarks, "RIGHT_KNEE")
        l_ankle = detection_body_part(landmarks, "LEFT_ANKLE")
        r_ankle = detection_body_part(landmarks, "RIGHT_ANKLE")

        # --- Compute angles ---
        left_arm_angle = body.angle_of_the_left_arm()
        right_arm_angle = body.angle_of_the_right_arm()
        left_leg_angle = body.angle_of_the_left_leg()
        right_leg_angle = body.angle_of_the_right_leg()
        abdomen_angle = body.angle_of_the_abdomen()

        avg_arm_angle = (left_arm_angle + right_arm_angle) / 2
        avg_leg_angle = (left_leg_angle + right_leg_angle) / 2

        # --- Derived spatial features ---
        shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
        hip_y = (l_hip[1] + r_hip[1]) / 2
        knee_y = (l_knee[1] + r_knee[1]) / 2
        elbow_y = (l_elbow[1] + r_elbow[1]) / 2
        wrist_y = (l_wrist[1] + r_wrist[1]) / 2
        ankle_y = (l_ankle[1] + r_ankle[1]) / 2

        # How vertical vs horizontal is the torso?
        torso_y_delta = abs(shoulder_y - hip_y)
        body_is_horizontal = torso_y_delta < 0.12
        body_is_vertical = torso_y_delta > 0.15

        scores = {}

        # ----- PUSH-UP -----
        s = 0.0
        # Body roughly horizontal (shoulders and hips at similar Y)
        if body_is_horizontal:
            s += 0.30
        elif torso_y_delta < 0.18:
            s += 0.15
        # Arm angle in push-up range (cycling between bent and straight)
        if avg_arm_angle < 160:
            s += 0.15
        if avg_arm_angle < 100:
            s += 0.15
        # Legs relatively straight
        if avg_leg_angle > 140:
            s += 0.15
        # Wrists roughly at or below shoulder height (supporting body on hands)
        if abs(wrist_y - shoulder_y) < 0.15:
            s += 0.10
        # Hips not far below shoulders (not standing)
        if torso_y_delta < 0.20:
            s += 0.10
        # Penalise if body is clearly vertical (standing)
        if body_is_vertical and shoulder_y < hip_y:
            s -= 0.20
        scores["push-up"] = max(0.0, min(s, 1.0))

        # ----- PULL-UP -----
        s = 0.0
        # Wrists above or near elbows (hands gripping bar overhead)
        if wrist_y < elbow_y:
            s += 0.25
        elif abs(wrist_y - elbow_y) < 0.05:
            s += 0.10
        # Elbows above or near shoulders
        if elbow_y < shoulder_y:
            s += 0.20
        elif abs(elbow_y - shoulder_y) < 0.05:
            s += 0.10
        # Nose rising to or above elbow level (pulling up)
        if nose[1] < elbow_y:
            s += 0.20
        # Body vertical
        if body_is_vertical:
            s += 0.15
        # Arms not fully straight (engaged in pull-up motion)
        if avg_arm_angle < 150:
            s += 0.10
        # Penalise if body is horizontal
        if body_is_horizontal:
            s -= 0.20
        # Penalise if wrists are below hips (not overhead)
        if wrist_y > hip_y:
            s -= 0.15
        scores["pull-up"] = max(0.0, min(s, 1.0))

        # ----- SIT-UP -----
        s = 0.0
        # Abdomen angle (shoulder–hip–knee) is the primary signal
        if abdomen_angle < 120:
            s += 0.25
        if abdomen_angle < 80:
            s += 0.15
        # Legs bent (knees bent, feet planted)
        if avg_leg_angle < 130:
            s += 0.15
        # Hips in the lower portion of the frame (lying/sitting on ground)
        if hip_y > 0.45:
            s += 0.10
        # Ankles should be relatively close to hips in Y (legs folded, not standing)
        if abs(ankle_y - hip_y) < 0.15:
            s += 0.10
        # Shoulders above hips (torso lifting up)
        if shoulder_y < hip_y:
            s += 0.10
        # Body NOT strongly vertical (distinguish from squat/standing)
        if body_is_horizontal or torso_y_delta < 0.20:
            s += 0.10
        # Penalise if legs are straight (standing or push-up)
        if avg_leg_angle > 160:
            s -= 0.15
        scores["sit-up"] = max(0.0, min(s, 1.0))

        # ----- SQUAT -----
        s = 0.0
        # Body vertical / upright posture
        if body_is_vertical:
            s += 0.20
        elif torso_y_delta > 0.12:
            s += 0.10
        # Leg angle in squat range (cycling 70°–160°)
        if avg_leg_angle < 150:
            s += 0.15
        if avg_leg_angle < 110:
            s += 0.15
        # Proper vertical ordering: shoulders above hips above knees
        if shoulder_y < hip_y and hip_y < knee_y:
            s += 0.15
        elif shoulder_y < hip_y:
            s += 0.05
        # Arms not supporting body (not push-up); arms typically forward or at sides
        if avg_arm_angle > 130:
            s += 0.05
        # Ankles below knees
        if ankle_y > knee_y:
            s += 0.10
        # Penalise if body is horizontal (push-up / sit-up)
        if body_is_horizontal:
            s -= 0.25
        # Penalise if legs are very straight AND no temporal movement (idle standing)
        if avg_leg_angle > 170:
            s -= 0.10
        scores["squat"] = max(0.0, min(s, 1.0))

        # ----- WALK -----
        walk_score = self._compute_walking_score(landmarks, body_is_vertical, avg_leg_angle)
        scores["walk"] = walk_score

        return scores

    def _compute_walking_score(self, landmarks, body_is_vertical, avg_leg_angle):
        """
        Walking requires temporal analysis — alternating knee displacement over
        multiple frames. A single static pose cannot confirm walking.
        """
        s = 0.0

        # Body must be upright
        if body_is_vertical:
            s += 0.15
        else:
            # Walking almost always has an upright torso
            return max(0.0, s)

        # Legs should not be deeply bent (not a squat)
        if 120 < avg_leg_angle < 178:
            s += 0.10
        elif avg_leg_angle <= 120:
            # Too bent — probably squatting
            return max(0.0, s - 0.10)

        # Temporal knee displacement analysis
        if len(self.knee_history) >= 10:
            recent = list(self.knee_history)[-15:]  # last ~0.5s of data

            # Measure how much the left and right knees are alternating
            l_knees = [p[0] for p in recent]
            r_knees = [p[1] for p in recent]

            # Calculate total absolute displacement of each knee
            l_displacement = sum(abs(l_knees[i] - l_knees[i - 1]) for i in range(1, len(l_knees)))
            r_displacement = sum(abs(r_knees[i] - r_knees[i - 1]) for i in range(1, len(r_knees)))

            total_displacement = l_displacement + r_displacement

            # Walking produces continuous lateral knee displacement
            if total_displacement > 0.15:
                s += 0.25
            elif total_displacement > 0.08:
                s += 0.15
            elif total_displacement > 0.03:
                s += 0.05

            # Check for alternating pattern — knees should cross each other
            crossings = 0
            for i in range(1, len(recent)):
                prev_diff = recent[i - 1][0] - recent[i - 1][1]  # left - right
                curr_diff = recent[i][0] - recent[i][1]
                if (prev_diff > 0 and curr_diff < 0) or (prev_diff < 0 and curr_diff > 0):
                    crossings += 1

            if crossings >= 3:
                s += 0.30
            elif crossings >= 2:
                s += 0.20
            elif crossings >= 1:
                s += 0.10
        else:
            # Not enough temporal data — can't confirm walking
            s *= 0.3  # heavily discount any walking score

        return max(0.0, min(s, 1.0))
