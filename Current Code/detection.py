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

    if image_filters.is_significant_tape_present(frame):
        print("Tape detected. Determining direction.")
        robot_api.api.v2.movement.stop()

        roi_frame_lanes, _ = roi_utils.apply_three_lane_roi_mask(frame, frame_width, frame_height)
        hsv_roi = cv2.cvtColor(roi_frame_lanes, cv2.COLOR_BGR2HSV)

        lower_path_color = np.array([90, 80, 40])
        upper_path_color = np.array([150, 255, 255])
        mask = image_filters.apply_color_filter(hsv_roi, lower_path_color, upper_path_color)
        masked = cv2.bitwise_and(roi_frame_lanes, roi_frame_lanes, mask=mask)

        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        detected_lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=50,
            minLineLength=max(30, frame_height // 10),
            maxLineGap=max(10, frame_height // 20)
        )

        filtered_lines, _ = line_processing_utils.filter_lines_by_angle(
            detected_lines, angle_threshold=25, previous_lines_state=[]
        )

        direction = "Middle"
        if filtered_lines:
            _, direction_state = line_processing_utils.draw_lines_and_interpret_lanes(
                frame, filtered_lines,
                line_detection_state.copy(),
                roi_utils.determine_line_roi_position
            )
            direction = direction_state.get("direction", "Middle")

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

    # Image Pre-processing for Line Detection
    lower_path_color = np.array([90, 80, 40])
    upper_path_color = np.array([150, 255, 255])
    roi_frame_lanes, _ = roi_utils.apply_three_lane_roi_mask(frame, frame_width, frame_height)
    hsv_roi_lanes = cv2.cvtColor(roi_frame_lanes, cv2.COLOR_BGR2HSV)
    color_mask_lanes = image_filters.apply_color_filter(hsv_roi_lanes, lower_path_color, upper_path_color)
    masked_by_color_lanes = cv2.bitwise_and(roi_frame_lanes, roi_frame_lanes, mask=color_mask_lanes)
    gray_lanes = cv2.cvtColor(masked_by_color_lanes, cv2.COLOR_BGR2GRAY)
    blurred_lanes = cv2.GaussianBlur(gray_lanes, (5, 5), 0)
    edges_lanes = cv2.Canny(blurred_lanes, 50, 150)

    # Line Detection (HoughLinesP)
    detected_lines = cv2.HoughLinesP(
        edges_lanes, rho=1, theta=np.pi / 180, threshold=50,
        minLineLength=max(30, frame_height // 10),
        maxLineGap=max(10, frame_height // 20)
    )

    # Filter, Process, and Draw Lines
    filtered_lines, line_detection_state["previous_lines"] = line_processing_utils.filter_lines_by_angle(
        detected_lines,
        angle_threshold=25,
        previous_lines_state=line_detection_state["previous_lines"]
    )

    line_image_overlay = np.zeros_like(frame)
    if filtered_lines:
        line_image_overlay, line_detection_state = line_processing_utils.draw_lines_and_interpret_lanes(
            frame, filtered_lines, line_detection_state, roi_utils.determine_line_roi_position
        )
        frame = cv2.addWeighted(frame, 0.8, line_image_overlay, 1, 0)

    current_driving_direction = line_detection_state.get("direction", "Middle")

    # Decision Making for Robot Movement (Path Following)
    if line_detection_state.get("vertical_detected"):
        action_flags['horizontal_line_processed'] = False

        if current_driving_direction == "Left" and not action_flags["turned_left"]:
            action_flags["CURRENTLY_RUNNING_PROCEDURE"] = True
            threading.Thread(target=robot_actions.execute_turn, args=("Left", action_flags, turn_parameters)).start()
        elif current_driving_direction == "Right" and not action_flags["turned_right"]:
            action_flags["CURRENTLY_RUNNING_PROCEDURE"] = True
            threading.Thread(target=robot_actions.execute_turn, args=("Right", action_flags, turn_parameters)).start()
        elif current_driving_direction == "Middle":
            action_flags["turned_left"] = False
            action_flags["turned_right"] = False
            robot_api.api.v2.movement.forward()
        else:
            robot_api.api.v2.movement.forward()
    else:
        robot_api.api.v2.movement.stop()
        action_flags["turned_left"] = False
        action_flags["turned_right"] = False

    #  Horizontal Line Detection for Intersections (T-junctions)
    if not action_flags["CURRENTLY_RUNNING_PROCEDURE"] and not action_flags["horizontal_line_processed"]:
        is_horizontal_present = line_detection_state.get("horizontal_detected", False)
        if is_horizontal_present:
            print("TRANSVERSE HORIZONTAL LINE DETECTED - Potential T-Junction.")
            robot_api.api.v2.movement.stop()
            action_flags["CURRENTLY_RUNNING_PROCEDURE"] = True
            threading.Thread(target=robot_actions.execute_intersection_turn,
                             args=(action_flags, turn_parameters, "Right")).start()
            return frame


    return frame
