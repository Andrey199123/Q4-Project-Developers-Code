import cv2
import numpy as np
import time
import threading
from gpiozero import DistanceSensor
import os

from robot_api.camera import orb_detector
from robot_api.camera import line_processing_utils  # This module is key now
from robot_api.camera import image_filters  # Still used for initial tape detection and saturation_mask
from robot_api.camera import robot_actions
import robot_api.api.v2.movement  # Direct import for explicit calls if any remain

previous_mid_x = None
alpha = 0.7  # Smoothing factor for exponential moving average


def saturation_mask(frame2, saturation_threshold=0.4, min_percent=10, min_region_size=600):
    frame = cv2.convertScaleAbs(frame2, alpha=1, beta=0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    total_pixels = frame.shape[0] * frame.shape[1]

    threshold_value = int(saturation_threshold * 255)
    binary_mask = (saturation > threshold_value).astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    filtered_mask = np.zeros_like(binary_mask)
    for i in range(1, num_labels):  # skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_region_size:
            filtered_mask[labels == i] = 255
    threshold_value = int(saturation_threshold * 255)
    saturated_pixels = np.sum(saturation > threshold_value)
    percent_saturated = (saturated_pixels / total_pixels) * 100
    if percent_saturated > 70:
        inverted_mask = cv2.bitwise_not(filtered_mask)
        return inverted_mask
    else:
        return filtered_mask



def smooth_value(new_value, previous_value, alpha=0.7):
    if previous_value is None:
        return new_value
    return int(alpha * previous_value + (1 - alpha) * new_value)


def separate_lines(lines):
    left_lines = []
    right_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Add small value to prevent division by zero
            intercept = y1 - slope * x1
            if slope < 0:  # Left lane
                left_lines.append((slope, intercept))
            elif slope > 0:  # Right lane
                right_lines.append((slope, intercept))

    left_lane = np.mean(left_lines, axis=0) if left_lines else None
    right_lane = np.mean(right_lines, axis=0) if right_lines else None

    return left_lane, right_lane


def draw_lane_lines(frame, left_lane, right_lane):
    global previous_mid_x

    line_image = np.zeros_like(frame)
    height = frame.shape[0]

    y_bottom = height
    y_top = int(height * .4)  # Draw up to 60% of the frame

    if left_lane is not None and right_lane is not None:
        left_slope, left_intercept = left_lane
        right_slope, right_intercept = right_lane

        left_x_bottom = int((y_bottom - left_intercept) / left_slope)
        right_x_bottom = int((y_bottom - right_intercept) / right_slope)

        left_x_top = int((y_top - left_intercept) / left_slope)
        right_x_top = int((y_top - right_intercept) / right_slope)

        cv2.line(line_image, (left_x_bottom, y_bottom), (left_x_top, y_top), (255, 0, 0), 3)  # Blue
        cv2.line(line_image, (right_x_bottom, y_bottom), (right_x_top, y_top), (255, 0, 0), 3)  # Blue

        mid_x_bottom = int((left_x_bottom + right_x_bottom) / 2)
        mid_x_top = int((left_x_top + right_x_top) / 2)

        if previous_mid_x is None:
            previous_mid_x = mid_x_bottom

        smoothed_mid_x_bottom = smooth_value(mid_x_bottom, previous_mid_x, alpha=0.8)
        previous_mid_x = smoothed_mid_x_bottom

        cv2.line(line_image, (smoothed_mid_x_bottom, y_bottom), (mid_x_top, y_top), (0, 0, 255), 3)  # Red

    elif left_lane is not None:
        slope, intercept = left_lane
        left_x1 = int((y_bottom - intercept) / slope)
        left_x2 = int((y_top - intercept) / slope)
        cv2.line(line_image, (left_x1, y_bottom), (left_x2, y_top), (255, 0, 0), 3)

    elif right_lane is not None:
        slope, intercept = right_lane
        right_x1 = int((y_bottom - intercept) / slope)
        right_x2 = int((y_top - intercept) / slope)
        cv2.line(line_image, (right_x1, y_bottom), (right_x2, y_top), (255, 0, 0), 3)

    return line_image


def apply_lane_detection(frame):
    try:
        processed_frame = frame.copy()

        mask = saturation_mask(processed_frame)

        masked_frame = cv2.bitwise_and(processed_frame, processed_frame, mask=mask)

        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blurred, 50, 150)

        lines = cv2.HoughLinesP(edges,
                                rho=1,
                                theta=np.pi / 180,
                                threshold=50,
                                minLineLength=40,
                                maxLineGap=10)

        if lines is not None:
            left_lane, right_lane = separate_lines(lines)
            lane_image = draw_lane_lines(processed_frame, left_lane, right_lane)
            processed_frame = cv2.addWeighted(processed_frame, 0.8, lane_image, 1, 0)

        return processed_frame

    except Exception as e:
        print(f"Error in lane detection: {e}")
        return frame


try:
    ultrasonic = DistanceSensor(trigger=27, echo=17, max_distance=2)  # Max distance in meters
    print("Ultrasonic sensor initialized.")
except Exception as e:
    print(f"Failed to initialize ultrasonic sensor: {e}")
    ultrasonic = None

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
OBSTACLE_DISTANCE_THRESHOLD_CM = 20
CURRENTLY_RUNNING_PROCEDURE = False
path_detected = False


def process_frame(frame):
    global action_flags, line_detection_state, turn_parameters, OBSTACLE_DISTANCE_THRESHOLD_CM
    global path_detected
    if frame is None:
        print("Received empty frame.")
        return None

    frame_height, frame_width = frame.shape[:2]
    # Martian Detection (unchanged)
    frame, action_flags["not_alone_timer_end"], martian_found = orb_detector.detect_martian_orb(
        frame, action_flags["not_alone_timer_end"]
    )
    if martian_found:
        robot_api.api.v2.movement.stop()
        return apply_lane_detection(frame)
    # Always check for obstacles first, even during procedures
    if ultrasonic:
        distance_cm = ultrasonic.distance * 100
        if distance_cm < OBSTACLE_DISTANCE_THRESHOLD_CM:
            print(f"Obstacle detected at {distance_cm:.1f} cm.")
            robot_api.api.v2.movement.stop()
            action_flags["object_detected"] = True
            action_flags["CURRENTLY_RUNNING_PROCEDURE"] = False  # Critical: Force-stop ongoing procedures
            threading.Thread(target=robot_actions.execute_obstacle_avoidance, args=(action_flags,)).start()
            return apply_lane_detection(frame)

    # Early return if procedure is running (after obstacle check)
    if action_flags["CURRENTLY_RUNNING_PROCEDURE"]:
        return apply_lane_detection(frame)

    # Rest of your existing path detection logic...
    if not path_detected:
        robot_api.api.v2.movement.forward()
        if image_filters.is_significant_tape_present(frame):
            print("PATH DETECTED")
            robot_api.api.v2.movement.stop()
            path_detected = True

    if path_detected:
        processed_mask_with_roi, updated_line_state = line_processing_utils.get_direction_from_roi_saturation(
            frame, line_detection_state
        )

        direction = updated_line_state["direction"]
        line_detection_state.update(updated_line_state)

        print(f"Detected Direction (ROI Saturation): {direction}")

        if not action_flags["CURRENTLY_RUNNING_PROCEDURE"]:
            action_flags["CURRENTLY_RUNNING_PROCEDURE"] = True
            threading.Thread(
                target=robot_actions.execute_turn, 
                args=(direction, action_flags, turn_parameters)
            ).start()

        final_display_frame = cv2.cvtColor(processed_mask_with_roi, cv2.COLOR_GRAY2BGR)
        _, roi_mask_for_viz = line_processing_utils.ROI(frame, frame_width, frame_height)
        colored_roi_overlay = np.zeros_like(final_display_frame, dtype=np.uint8)
        colored_roi_overlay[roi_mask_for_viz == 255] = [0, 255, 0]
        final_display_frame = cv2.addWeighted(final_display_frame, 1.0, colored_roi_overlay, 0.3, 0)

        return apply_lane_detection(frame)

    saturated_bgr = cv2.cvtColor(saturated_frame_mask, cv2.COLOR_GRAY2BGR)
    return apply_lane_detection(frame)
