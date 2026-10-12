
import mediapipe as mp
import pandas as pd
import numpy as np
import cv2

# Lazy-load mp_pose to avoid import issues
_mp_pose = None

def get_mp_pose():
    global _mp_pose
    if _mp_pose is None:
        _mp_pose = mp.solutions.pose
    return _mp_pose

def calculate_angle(a, b, c):
    a = np.array(a)  
    b = np.array(b)  
    c = np.array(c)  

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) -\
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle



def detection_body_part(landmarks, body_part_name):
    mp_pose = get_mp_pose()
    return [
        landmarks[mp_pose.PoseLandmark[body_part_name].value].x,
        landmarks[mp_pose.PoseLandmark[body_part_name].value].y,
        landmarks[mp_pose.PoseLandmark[body_part_name].value].visibility
    ]


def detection_body_parts(landmarks):
    mp_pose = get_mp_pose()
    body_parts = pd.DataFrame(columns=["body_part", "x", "y"])

    for i, lndmrk in enumerate(mp_pose.PoseLandmark):
        lndmrk = str(lndmrk).split(".")[1]
        cord = detection_body_part(landmarks, lndmrk)
        body_parts.loc[i] = lndmrk, cord[0], cord[1]

    return body_parts


def score_table(exercise, frame, counter, status, counters=None,
                detection_mode="MANUAL", confidence=0.0, is_transitioning=False):
    overlay = frame.copy()
    
    # Top overlay bar for Active Exercise info (taller to fit detection info)
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 140), (20, 20, 20), -1)
    
    # Right summary box if counters dictionary is provided
    if counters:
        cv2.rectangle(overlay, (frame.shape[1] - 220, 0), (frame.shape[1], 190), (30, 30, 30), -1)
    
    # Bottom menu bar
    cv2.rectangle(overlay, (0, frame.shape[0] - 40), (frame.shape[1], frame.shape[0]), (20, 20, 20), -1)
    
    # Blend overlay with original frame for semi-transparency
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Active exercise details with AUTO/MANUAL tag
    activity_name = exercise.upper().replace("-", " ")
    mode_tag = f"({detection_mode})"
    status_str = "READY / UP" if status else "IN MOTION / DOWN"
    
    # Mode line — colour-coded: cyan for AUTO, yellow for MANUAL
    mode_color = (0, 255, 255) if detection_mode == "AUTO" else (0, 255, 255)
    cv2.putText(frame, f"MODE: {activity_name} {mode_tag}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2, cv2.LINE_AA)

    # DETECTING... indicator (pulsing orange) when transitioning
    if is_transitioning:
        cv2.putText(frame, "DETECTING...", (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
        reps_y = 80
    else:
        reps_y = 60

    # Reps counter
    cv2.putText(frame, f"REPS: {counter}", (15, reps_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2, cv2.LINE_AA)

    # Status line
    cv2.putText(frame, f"STATUS: {status_str}", (15, reps_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # Detection info line
    conf_pct = int(confidence * 100)
    det_color = (0, 255, 0) if conf_pct >= 60 else (0, 200, 255) if conf_pct >= 35 else (0, 0, 255)
    cv2.putText(frame, f"DETECTION: {detection_mode}  |  CONFIDENCE: {conf_pct}%",
                (15, reps_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, det_color, 1, cv2.LINE_AA)

    # Render summary table on top right
    if counters:
        cv2.putText(frame, "WORKOUT SUMMARY", (frame.shape[1] - 210, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (frame.shape[1] - 210, 32), (frame.shape[1] - 10, 32), (150, 150, 150), 1)
        
        ex_keys = [("push-up", "1: Push-Up"), ("pull-up", "2: Pull-Up"), ("sit-up", "3: Sit-Up"), ("squat", "4: Squat"), ("walk", "5: Walk")]
        for idx, (ex_key, label) in enumerate(ex_keys):
            y_pos = 55 + (idx * 25)
            cnt = counters.get(ex_key, 0)
            color = (0, 255, 255) if ex_key == exercise else (200, 200, 200)
            cv2.putText(frame, f"{label}: {cnt}", (frame.shape[1] - 210, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2 if ex_key == exercise else 1, cv2.LINE_AA)

    # Render hotkeys menu at the bottom (includes [A]uto)
    hotkey_text = "[1-5]Manual  [A]uto  |  [R]eset  [C]lear All  [Q]uit"
    cv2.putText(frame, hotkey_text, (15, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return frame

