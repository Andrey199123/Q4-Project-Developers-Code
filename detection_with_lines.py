# robot_api/camera/detection.py
import cv2
import numpy as np
import time
import threading
from gpiozero import DistanceSensor
import os

# Import from our new modules
from robot_api.camera import orb_detector
from robot_api.camera import line_processing_utils  # This module is key now
from robot_api.camera import image_filters  # Still used for initial tape detection and saturation_mask
from robot_api.camera import robot_actions
import robot_api.api.v2.movement  # Direct import for explicit calls if any remain

# --- Hardware Initialization ---
try:
    ultrasonic = DistanceSensor(trigger=27, echo=17, max_distance=2)  # Max distance in meters
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
    "horizontal_line_processed": False,  # This might need re-evaluation if horizontal detection changes
    "initial_blue_found": False,
}

line_detection_state = {
    "previous_lines": [],  # No longer used for line detection, can be removed or left as placeholder
    "previous_mid_x": None,  # No longer used for line detection, can be removed or left as placeholder
    "vertical_detected": False,  # Will now reflect if 'white' pixels were found in left/right ROIs
    "horizontal_detected": False,  # Will need a new method if still required, or remove
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


def compute_line(lines, wack):
    used = [False] * len(lines)
    ans = []

    for i in range(len(lines)):
        if used[i] == False:
            used[i] = True
            x1, y1, x2, y2 = lines[i][0]
            base_slope = compute_slope(lines[i][0])
            if base_slope < 0.1 and base_slope > -0.1:
                for j in range(i, len(lines)):
                    temp = compute_slope(lines[j][0])
                    if (
                        used[j] == False
                        and temp < 0.1
                        and temp > -0.1
                        and lines[j][0][1] < y1 + 40
                        and lines[j][0][1] > y1 - 40
                    ):
                        x1, y1, x2, y2 = update(
                            lines[j][0], x1, y1, x2, y2, temp
                        )
                        used[j] = True

                ans.append([x1, int((y1 + y2) / 2), x2, int((y1 + y2) / 2)])
            else:
                for j in range(i, len(lines)):
                    temp = compute_slope(lines[j][0])

                    if (
                        used[j] == False
                        and temp < base_slope + 0.20
                        and temp > base_slope - 0.20
                    ):
                        x1, y1, x2, y2 = update(
                            lines[j][0], x1, y1, x2, y2, temp
                        )
                        used[j] = True
                ans.append([x1, y1, x2, y2])

    return ans


def compute_center(lines, frame):
    neg = [0, 0, 0, 0]
    pos = [0, 0, 0, 0]

    for line in lines:
        if compute_slope(line) > -2 and compute_length(line) > compute_length(
            pos
        ):
            pos = line.copy()
        elif compute_slope(line) < 2 and compute_length(line) > compute_length(
            neg
        ):
            neg = line.copy()

    positive_slope = compute_slope(pos)
    negative_slope = compute_slope(neg)
    if (
        abs(negative_slope) > positive_slope - 3.5
        and abs(negative_slope) < positive_slope + 3.5
        and compute_length(pos) > 100
        and compute_length(neg) > 100
        and min(neg[0], neg[2]) < min(pos[0], pos[2])
    ):
        if neg[3] > pos[1]:
            if negative_slope != 0:
                neg[2] = int(neg[2] - (neg[3] - pos[1]) / negative_slope)
            neg[3] = pos[1]
        else:
            if positive_slope != 0:
                pos[0] = int(pos[0] + (neg[3] - pos[1]) / positive_slope)
            pos[1] = neg[3]

        if neg[1] < pos[3]:
            if negative_slope != 0:
                neg[0] = int(neg[0] - (neg[1] - pos[3]) / negative_slope)
            neg[1] = pos[3]
        else:
            if positive_slope != 0:
                pos[2] = int(pos[2] + (neg[1] - pos[3]) / positive_slope)
            pos[3] = neg[1]
        temp = int(
            (int((pos[2] + neg[0]) / 2) + int((pos[0] + neg[2]) / 2)) / 2
        )
        if min(neg[0], neg[2]) < min(pos[0], pos[2]):
            cv2.line(frame, (pos[0], pos[1]), (pos[2], pos[3]), (0, 255, 0), 10)
            cv2.line(frame, (neg[0], neg[1]), (neg[2], neg[3]), (0, 255, 0), 10)
            return [temp + 1, pos[1], temp, pos[3]]
        else:
            return [0, 0, 0, 0]
    else:
        return [0, 0, 0, 0]

#For Threading, to make the movement work in a separate Thread in this file
CURRENTLY_RUNNING_PROCEDURE = False


def process_frame(frame):
    global action_flags, line_detection_state, turn_parameters, OBSTACLE_DISTANCE_THRESHOLD_CM
    global path_detected
    if frame is None:
        print("Received empty frame.")
        return None

    frame_height, frame_width = frame.shape[:2]

    # --- Obstacle Detection ---
    if ultrasonic and not action_flags["CURRENTLY_RUNNING_PROCEDURE"]:
        distance_cm = ultrasonic.distance * 100
        if distance_cm < OBSTACLE_DISTANCE_THRESHOLD_CM:
            print(f"Obstacle detected at {distance_cm:.1f} cm.")
            robot_api.api.v2.movement.stop()
            action_flags["object_detected"] = True
            threading.Thread(target=robot_actions.execute_obstacle_avoidance, args=(action_flags,)).start()
            return frame

            # Initial saturation mask for general display/debug if needed, or for initial path detection
    saturated_frame_mask = image_filters.saturation_mask(frame)

    if action_flags["CURRENTLY_RUNNING_PROCEDURE"]:
        return cv2.cvtColor(saturated_frame_mask, cv2.COLOR_GRAY2BGR)  # Return for consistent display

    # --- Path Detection Logic (initial tape presence) ---
    if not path_detected:
        robot_api.api.v2.movement.forward()
        # image_filters.is_significant_tape_present likely uses saturation logic, so this is compatible
        if image_filters.is_significant_tape_present(frame):
            print("PATH DETECTED")
            robot_api.api.v2.movement.stop()
            path_detected = True
            # Continue to direction determination in the same frame

    if path_detected:
        # --- NEW: Get direction directly from ROI saturation ---
        # This function now encapsulates the ROI and saturation_mask logic
        # It returns the saturation mask (with ROI applied) and updates line_detection_state
        processed_mask_with_roi, updated_line_state = line_processing_utils.get_direction_from_roi_saturation(
            frame, line_detection_state  # Pass the original frame and the state dict
        )

        direction = updated_line_state["direction"]
        line_detection_state.update(updated_line_state)  # Update global state

        print(f"Detected Direction (ROI Saturation): {direction}")

        if not action_flags["CURRENTLY_RUNNING_PROCEDURE"]:
            action_flags["CURRENTLY_RUNNING_PROCEDURE"] = True
            threading.Thread(target=robot_actions.execute_turn, args=(direction, action_flags, turn_parameters)).start()
        final_display_frame = cv2.cvtColor(processed_mask_with_roi, cv2.COLOR_GRAY2BGR)
        _, roi_mask_for_viz = line_processing_utils.ROI(frame, frame_width, frame_height)
        colored_roi_overlay = np.zeros_like(final_display_frame, dtype=np.uint8)
        colored_roi_overlay[roi_mask_for_viz == 255] = [0, 255, 0]  # Green color for ROI
        final_display_frame = cv2.addWeighted(final_display_frame, 1.0, colored_roi_overlay, 0.3, 0)
        # 1. Compute lines on the ROI-masked saturated frame
        n_lines, centerline = line_processing_utils.detect_lines_in_mask(
            processed_mask_with_roi, min_line_length=100, max_line_gap=30, angle_range=(0, 180)
        )

        # 2. Draw lines with your logic
        if n_lines is not None:
            for line in n_lines:
                x1, y1, x2, y2 = line
                if -0.1 < line_processing_utils.compute_slope(line) < 0.1:
                    continue
                elif line_processing_utils.compute_length(line) > 200:
                    continue
                cv2.line(final_display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw line in green

        # 3. Draw the centerline in blue if it's meaningful
        if centerline and line_processing_utils.compute_slope(centerline) != 0:
            x1, y1, x2, y2 = centerline
            cv2.line(final_display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue line for centerline
        return final_display_frame  # Return the saturation mask with ROI overlayed

    # Martian Detection
    frame, action_flags["not_alone_timer_end"], martian_found = orb_detector.detect_martian_orb(
        frame, action_flags["not_alone_timer_end"]
    )
    if martian_found:
        robot_api.api.movement.v2.stop()
        return frame
    return cv2.cvtColor(saturated_frame_mask, cv2.COLOR_GRAY2BGR)
