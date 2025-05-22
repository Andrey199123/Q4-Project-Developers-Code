# robot_api/camera/detection.py
import os
import time
import threading
import math

import cv2
import numpy as np
from gpiozero import DistanceSensor

# Import from our modules
from robot_api.camera import orb_detector
from robot_api.camera import line_processing_utils  # ROI & direction logic
from robot_api.camera import image_filters
from robot_api.camera import robot_actions
import robot_api.api.v2.movement

# --- Hardware Initialization ---
try:
    ultrasonic = DistanceSensor(trigger=27, echo=17, max_distance=2)
    print("Ultrasonic sensor initialized.")
except Exception as e:
    print(f"Failed to initialize ultrasonic sensor: {e}")
    ultrasonic = None

# --- Global State Variables ---
action_flags = {
    "CURRENTLY_RUNNING_PROCEDURE": False,
    "object_detected": False,
    "turned_left": False,
    "turned_right": False,
    "not_alone_timer_end": 0.0,
    "horizontal_line_processed": False,
    "initial_blue_found": False,
}

line_detection_state = {
    "previous_lines": [],
    "previous_mid_x": None,
    "vertical_detected": False,
    "horizontal_detected": False,
    "direction": "Middle",
}

turn_parameters = {
    "left_fwd_time": 1.0,
    "left_turn_time": 1.0,
    "right_fwd_time": 1.0,
    "right_turn_time": 1.0,
    "intersection_fwd_time": 1.8,
    "intersection_turn_time": 1.1,
    "intersection_exit_fwd_time": 1.0
}
OBSTACLE_DISTANCE_THRESHOLD_CM = 25
path_detected = False

# --- Line Drawing Utilities (from old detection code) ---
def compute_slope(line):
    x1, y1, x2, y2 = line
    if x2 - x1 == 0:
        return 0
    return (y2 - y1) / (x2 - x1)


def compute_length(line):
    x1, y1, x2, y2 = line
    return np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)


def update(line, x1, y1, x2, y2, slope):
    x1 = min(x1, line[0])
    x2 = max(x2, line[2])
    if slope > 0:
        y2 = max(y2, line[3])
        y1 = min(y1, line[1])
    else:
        y2 = min(y2, line[3])
        y1 = max(y1, line[1])
    return x1, y1, x2, y2


def compute_line(lines):
    used = [False] * len(lines)
    merged = []
    for i in range(len(lines)):
        if not used[i]:
            used[i] = True
            x1, y1, x2, y2 = lines[i][0]
            base_slope = compute_slope(lines[i][0])
            for j in range(i + 1, len(lines)):
                temp_slope = compute_slope(lines[j][0])
                if not used[j] and abs(temp_slope - base_slope) < 0.20:
                    x1, y1, x2, y2 = update(lines[j][0], x1, y1, x2, y2, temp_slope)
                    used[j] = True
            merged.append([x1, y1, x2, y2])
    return merged


def compute_center(lines, frame):
    neg = [0, 0, 0, 0]
    pos = [0, 0, 0, 0]
    for line in lines:
        slope = compute_slope(line)
        length = compute_length(line)
        if slope > 0.5 and length > compute_length(pos):
            pos = line.copy()
        elif slope < -0.5 and length > compute_length(neg):
            neg = line.copy()
    # draw best positive and negative lines
    if neg[3] != 0 and pos[3] != 0:
        cv2.line(frame, (pos[0], pos[1]), (pos[2], pos[3]), (0, 255, 0), 2)
        cv2.line(frame, (neg[0], neg[1]), (neg[2], neg[3]), (0, 255, 0), 2)
        # compute and draw center line between them
        center_x = int((pos[2] + neg[0]) / 2)
        center_y1 = int((pos[1] + neg[3]) / 2)
        center_y2 = int((pos[3] + neg[1]) / 2)
        cv2.line(frame, (center_x, center_y1), (center_x, center_y2), (255, 0, 0), 2)
        return [center_x, center_y1, center_x, center_y2]
    return [0, 0, 0, 0]


def process_frame(frame):
    global action_flags, line_detection_state, turn_parameters
    global OBSTACLE_DISTANCE_THRESHOLD_CM, path_detected

    if frame is None:
        print("Received empty frame.")
        return None

    h, w = frame.shape[:2]
    # --- Obstacle Detection ---
    if ultrasonic and not action_flags["CURRENTLY_RUNNING_PROCEDURE"]:
        dist = ultrasonic.distance * 100
        if dist < OBSTACLE_DISTANCE_THRESHOLD_CM:
            print(f"Obstacle detected at {dist:.1f} cm.")
            robot_api.api.v2.movement.stop()
            action_flags["object_detected"] = True
            threading.Thread(
                target=robot_actions.execute_obstacle_avoidance,
                args=(action_flags,)
            ).start()
            return frame

    # Initial saturation mask
    sat_mask = image_filters.saturation_mask(frame)
    if action_flags["CURRENTLY_RUNNING_PROCEDURE"]:
        return cv2.cvtColor(sat_mask, cv2.COLOR_GRAY2BGR)

    # --- Path Detection ---
    if not path_detected:
        robot_api.api.v2.movement.forward()
        if image_filters.is_significant_tape_present(frame):
            print("PATH DETECTED")
            robot_api.api.v2.movement.stop()
            path_detected = True

    # --- Direction & Line Drawing ---
    if path_detected:
        processed_mask, updated_state = line_processing_utils.get_direction_from_roi_saturation(frame, line_detection_state)
        line_detection_state.update(updated_state)
        direction = updated_state["direction"]
        print(f"Detected Direction: {direction}")
        if not action_flags["CURRENTLY_RUNNING_PROCEDURE"]:
            action_flags["CURRENTLY_RUNNING_PROCEDURE"] = True
            threading.Thread(
                target=robot_actions.execute_turn,
                args=(direction, action_flags, turn_parameters)
            ).start()
        # Prepare display frame
        display = cv2.cvtColor(processed_mask, cv2.COLOR_GRAY2BGR)
        # Overlay ROI boundary
        _, roi = line_processing_utils.ROI(frame, w, h)
        overlay = np.zeros_like(display)
        overlay[roi == 255] = (0, 255, 0)
        display = cv2.addWeighted(display, 1.0, overlay, 0.3, 0)
        # --- Draw Hough lines ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=60, maxLineGap=20)
        if lines is not None:
            merged = compute_line(lines)
            _ = compute_center(merged, display)
            for ln in merged:
                x1, y1, x2, y2 = ln
                cv2.line(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return display

    # --- Martian Detection ---
    frame, action_flags["not_alone_timer_end"], found = orb_detector.detect_martian_orb(
        frame, action_flags["not_alone_timer_end"]
    )
    if found:
        robot_api.api.v2.movement.stop()
        return frame

    # Default return
    return cv2.cvtColor(sat_mask, cv2.COLOR_GRAY2BGR)
