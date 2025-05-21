# robot_api/camera/detection.py
import cv2
import numpy as np
import time
import threading
from gpiozero import DistanceSensor
import os

# Import from our new modules
from robot_api.camera import orb_detector
from robot_api.camera import roi_utils
from robot_api.camera import line_processing_utils
from robot_api.camera import image_filters
from robot_api.camera import robot_actions
import robot_api.api.v2.movement # Direct import for explicit calls if any remain

# --- Hardware Initialization ---
try:
    ultrasonic = DistanceSensor(trigger=27, echo=17, max_distance=2) # Max distance in meters
    print("Ultrasonic sensor initialized.")
except Exception as e:
    print(f"Failed to initialize ultrasonic sensor: {e}")
    ultrasonic = None

# --- Global State Variables for robot behavior and processing ---
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
CURRENTLY_RUNNING_PROCEDURE = False
path_detected = False
def process_frame(frame):
    global action_flags, line_detection_state, turn_parameters, OBSTACLE_DISTANCE_THRESHOLD_CM

    if frame is None:
        print("Received empty frame.")
        return None

    frame_height, frame_width = frame.shape[:2]
    if ultrasonic and not action_flags["CURRENTLY_RUNNING_PROCEDURE"]:
        distance_cm = ultrasonic.distance * 100
        if distance_cm < OBSTACLE_DISTANCE_THRESHOLD_CM:
            print(f"Obstacle detected at {distance_cm:.1f} cm.")
            robot_api.api.v2.movement.stop()
            action_flags["object_detected"] = True
            threading.Thread(target=robot_actions.execute_obstacle_avoidance, args=(action_flags,)).start()
            return frame

    if action_flags["CURRENTLY_RUNNING_PROCEDURE"]:
        return frame  # Skip processing if already executing a procedure
    if not path_detected:
        robot_api.api.v2.movement.forward()
    if image_filters.is_significant_tape_present(frame):
        print("PATH DETECTED")
        robot_api.api.v2.movement.stop()
        path_detected = True
    if path_detected:
        direction = "Middle"
        # direction = draw_and_interpret_lanes(frame)
        print(f"Path Direction Detected: {direction}")
        action_flags["CURRENTLY_RUNNING_PROCEDURE"] = True
        threading.Thread(target=robot_actions.execute_turn, args=(direction, action_flags, turn_parameters)).start()
        return frame

    #Martian Detection
    frame, action_flags["not_alone_timer_end"], martian_found = orb_detector.detect_martian_orb(
        frame, action_flags["not_alone_timer_end"]
    )
    if martian_found:
        return frame

    # Obstacle Detection (Ultrasonic)
    if ultrasonic:
        distance_cm = ultrasonic.distance * 100
        if distance_cm < OBSTACLE_DISTANCE_THRESHOLD_CM:
            action_flags["object_detected"] = True
            if not action_flags["CURRENTLY_RUNNING_PROCEDURE"] and not martian_found:
                print(f"Main Phase: Obstacle detected at {distance_cm:.1f} cm. Threshold: {OBSTACLE_DISTANCE_THRESHOLD_CM} cm")
                robot_api.api.v2.movement.stop()
                threading.Thread(target=robot_actions.execute_obstacle_avoidance, args=(action_flags,)).start()
                return frame
        else:
            if action_flags["object_detected"]:
                print(f"Main Phase: Obstacle cleared (distance: {distance_cm:.1f} cm)")
            action_flags["object_detected"] = False

    if action_flags["CURRENTLY_RUNNING_PROCEDURE"] or martian_found:
        return frame
    frame = image_filters.saturation_mask(frame)
    return frame
