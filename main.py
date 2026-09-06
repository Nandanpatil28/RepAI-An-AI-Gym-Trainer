import cv2
import os
import argparse
from utils import *
import mediapipe as mp
from body_part_angle import BodyPartAngle
from types_of_exercise import TypeOfExercise
from exercise_classifier import ExerciseClassifier


ap = argparse.ArgumentParser(description="AI Fitness Trainer with Automatic Exercise Detection")
ap.add_argument("-t",
                "--exercise_type",
                type=str,
                default=None,
                help='Initial exercise mode: push-up, pull-up, sit-up, squat, walk (omit for AUTO detection)',
                required=False)
ap.add_argument("-vs",
                "--video_source",
                type=str,
                help='Sample video file name or path (leave empty to use live webcam)',
                required=False)
args = vars(ap.parse_args())

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Mapping keyboard keys 1-5 to exercise modes
exercise_map = {
    ord('1'): "push-up",
    ord('2'): "pull-up",
    ord('3'): "sit-up",
    ord('4'): "squat",
    ord('5'): "walk"
}

ALL_EXERCISES = ["push-up", "pull-up", "sit-up", "squat", "walk"]

# State dictionary to maintain counters and statuses for all exercises
counters = {ex: 0 for ex in ALL_EXERCISES}
statuses = {ex: True for ex in ALL_EXERCISES}

# Determine initial mode: AUTO (no -t flag) or MANUAL (with -t flag)
if args["exercise_type"] is not None:
    current_exercise = args["exercise_type"].lower() if args["exercise_type"].lower() in counters else "push-up"
    auto_mode = False
else:
    current_exercise = "push-up"
    auto_mode = True

using_webcam = args["video_source"] is None

# Initialise the exercise classifier
classifier = ExerciseClassifier(buffer_size=45, switch_threshold=30, min_confidence=0.35)
if not auto_mode:
    classifier.set_exercise(current_exercise)

# Detection state for HUD
detection_confidence = 0.0
is_transitioning = False


def load_video(video_name_or_path):
    if not video_name_or_path:
        return cv2.VideoCapture(0)
    if os.path.exists(video_name_or_path):
        return cv2.VideoCapture(video_name_or_path)
    folder_path = os.path.join("Exercise_videos", video_name_or_path)
    if os.path.exists(folder_path):
        return cv2.VideoCapture(folder_path)
    return cv2.VideoCapture(video_name_or_path)


cap = load_video(args["video_source"])
cap.set(3, 800)  # width
cap.set(4, 480)  # height

# setup mediapipe pose detection
with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while True:
        if not cap.isOpened():
            break

        ret, frame = cap.read()
        if not ret:
            # Loop sample video back to start when reaching the end
            if not using_webcam:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        frame = cv2.resize(frame, (800, 480), interpolation=cv2.INTER_AREA)
        # recolor frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        # make detection
        results = pose.process(frame)
        # recolor back to BGR
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # --- Automatic exercise detection ---
            if auto_mode:
                detected, confidence, transitioning = classifier.update(landmarks)
                detection_confidence = confidence
                is_transitioning = transitioning

                if detected is not None and detected != current_exercise:
                    current_exercise = detected

            # --- Rep counting (always runs for the active exercise) ---
            cnt, stat = TypeOfExercise(landmarks).calculate_exercise(
                current_exercise, counters[current_exercise], statuses[current_exercise])
            counters[current_exercise] = cnt
            statuses[current_exercise] = stat

        except Exception:
            pass

        # Determine display mode string
        detection_mode = "AUTO" if auto_mode else "MANUAL"

        # Render integrated HUD
        frame = score_table(
            current_exercise, frame,
            counters[current_exercise], statuses[current_exercise],
            counters=counters,
            detection_mode=detection_mode,
            confidence=detection_confidence,
            is_transitioning=is_transitioning
        )

        # render detections (for landmarks)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255),
                                       thickness=2,
                                       circle_radius=2),
                mp_drawing.DrawingSpec(color=(174, 139, 45),
                                       thickness=2,
                                       circle_radius=2),
            )

        cv2.imshow('AI Fitness Trainer', frame)
        key = cv2.waitKey(10) & 0xFF

        # --- Key handling ---
        if key == ord('q') or key == ord('Q'):
            break

        elif key in exercise_map:
            # Manual override: switch to MANUAL mode
            new_exercise = exercise_map[key]
            current_exercise = new_exercise
            auto_mode = False
            classifier.set_exercise(new_exercise)
            detection_confidence = 1.0
            is_transitioning = False
            # If running video mode, switch to matching sample video
            if not using_webcam:
                sample_video = f"{current_exercise}.mp4"
                if os.path.exists(os.path.join("Exercise_videos", sample_video)):
                    cap.release()
                    cap = cv2.VideoCapture(os.path.join("Exercise_videos", sample_video))

        elif key == ord('a') or key == ord('A'):
            # Toggle to AUTO detection mode
            auto_mode = True
            classifier.reset()
            is_transitioning = False

        elif key == ord('r') or key == ord('R'):
            counters[current_exercise] = 0
            statuses[current_exercise] = True

        elif key == ord('c') or key == ord('C'):
            for ex in counters:
                counters[ex] = 0
                statuses[ex] = True

    cap.release()
    cv2.destroyAllWindows()
