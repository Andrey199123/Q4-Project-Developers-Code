# robot_api/camera/detection.py
import cv2
import numpy as np
import time
import threading
from gpiozero import DistanceSensor
import os

# Import from our new modules
from robot_api.camera import orb_detector
# from robot_api.camera import roi_utils # No longer directly needed if ROI logic is fully in line_processing_utils
from robot_api.camera import line_processing_utils # This module is key now
from robot_api.camera import image_filters # Still used for initial tape detection and saturation_mask
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
    "horizontal_line_processed": False, # This might need re-evaluation if horizontal detection changes
    "initial_blue_found": False,
}

line_detection_state = {
    "previous_lines": [], # No longer used for line detection, can be removed or left as placeholder
    "previous_mid_x": None, # No longer used for line detection, can be removed or left as placeholder
    "vertical_detected": False, # Will now reflect if 'white' pixels were found in left/right ROIs
    "horizontal_detected": False, # Will need a new method if still required, or remove
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
        return cv2.cvtColor(saturated_frame_mask, cv2.COLOR_GRAY2BGR) # Return for consistent display
    
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
            frame, line_detection_state # Pass the original frame and the state dict
        )
        
        direction = updated_line_state["direction"]
        line_detection_state.update(updated_line_state) # Update global state
        
        print(f"Detected Direction (ROI Saturation): {direction}")

        if not action_flags["CURRENTLY_RUNNING_PROCEDURE"]:
            action_flags["CURRENTLY_RUNNING_PROCEDURE"] = True
            threading.Thread(target=robot_actions.execute_turn, args=(direction, action_flags, turn_parameters)).start()
        
        # --- Prepare the frame to return ---
        # processed_mask_with_roi is a 1-channel binary mask.
        # Convert it to 3-channel BGR so you can potentially draw on it or display it.
        final_display_frame = cv2.cvtColor(processed_mask_with_roi, cv2.COLOR_GRAY2BGR)

        # OPTIONAL: Draw the ROI boundary on this final frame for visualization
        # You would need to get the ROI mask again here, or have get_direction_from_roi_saturation return it.
        # For simplicity, let's assume `get_direction_from_roi_saturation` returns a mask where ROI is highlighted.
        # If not, you'd perform the ROI drawing here:
        _, roi_mask_for_viz = line_processing_utils.ROI(frame, frame_width, frame_height)
        colored_roi_overlay = np.zeros_like(final_display_frame, dtype=np.uint8)
        colored_roi_overlay[roi_mask_for_viz == 255] = [0, 255, 0] # Green color for ROI
        final_display_frame = cv2.addWeighted(final_display_frame, 1.0, colored_roi_overlay, 0.3, 0)


        return final_display_frame # Return the saturation mask with ROI overlayed

    # Martian Detection
    frame, action_flags["not_alone_timer_end"], martian_found = orb_detector.detect_martian_orb(
        frame, action_flags["not_alone_timer_end"]
    )
    if martian_found:
        return frame 
    if action_flags["CURRENTLY_RUNNING_PROCEDURE"] or martian_found:
        return frame
    
    # Default case: if no path detected, no obstacle, no martian, and no procedure running.
    # Return the saturated frame (converted to BGR for consistent display).
    return cv2.cvtColor(saturated_frame_mask, cv2.COLOR_GRAY2BGR)
