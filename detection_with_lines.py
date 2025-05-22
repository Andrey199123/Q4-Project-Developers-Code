import cv2
import numpy as np
import time
import threading
from gpiozero import DistanceSensor
import os

# Import from our new modules
from robot_api.camera import orb_detector
from robot_api.camera import line_processing_utils
from robot_api.camera import image_filters
from robot_api.camera import robot_actions
import robot_api.api.v2.movement

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
        # If a procedure is running, we still want to display a frame.
        # Let's ensure it's still based on the latest saturated mask or original frame with overlay.
        # For consistency, if CURRENTLY_RUNNING_PROCEDURE is true, we should probably
        # return a display frame that shows current robot state/path, not just a static mask.
        # For now, keeping it as before, but note this could be improved for better visual feedback.
        return cv2.cvtColor(saturated_frame_mask, cv2.COLOR_GRAY2BGR)
    
    # --- Path Detection Logic (initial tape presence) ---
    if not path_detected:
        robot_api.api.v2.movement.forward()
        if image_filters.is_significant_tape_present(frame):
            print("PATH DETECTED")
            robot_api.api.v2.movement.stop()
            path_detected = True
            # Continue to direction determination in the same frame

    if path_detected:
        # --- Get direction directly from ROI saturation ---
        processed_mask_with_roi, updated_line_state = line_processing_utils.get_direction_from_roi_saturation(
            frame, line_detection_state
        )
        
        direction = updated_line_state["direction"]
        line_detection_state.update(updated_line_state)
        
        print(f"Detected Direction (ROI Saturation): {direction}")

        if not action_flags["CURRENTLY_RUNNING_PROCEDURE"]:
            action_flags["CURRENTLY_RUNNING_PROCEDURE"] = True
            threading.Thread(target=robot_actions.execute_turn, args=(direction, action_flags, turn_parameters)).start()
        
        # --- Prepare the frame to return: Start with the saturated mask as base ---
        final_display_frame = cv2.cvtColor(processed_mask_with_roi, cv2.COLOR_GRAY2BGR)

        # OPTIONAL: Draw the ROI boundary on this final frame for visualization
        _, roi_mask_for_viz = line_processing_utils.ROI(frame, frame_width, frame_height)
        colored_roi_overlay = np.zeros_like(final_display_frame, dtype=np.uint8)
        colored_roi_overlay[roi_mask_for_viz == 255] = [0, 255, 0] # Green color for ROI
        final_display_frame = cv2.addWeighted(final_display_frame, 1.0, colored_roi_overlay, 0.3, 0)

        # --- Re-introducing Line Drawing for Visualization Only ---
        # Convert original frame to grayscale for Canny
        gray_for_lines = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply ROI specific to the old line detection if desired, or use full frame
        # For simplicity, let's use the full frame's edges but you can narrow it down.
        # If you want to use the specific blue color mask from the old 'detect_horizontal_line',
        # you'd need to re-implement that masking here first.
        
        # Applying a blur and Canny edge detector
        blurred_for_lines = cv2.GaussianBlur(gray_for_lines, (5, 5), 0)
        edges_for_lines = cv2.Canny(blurred_for_lines, 50, 150) # Canny parameters from old code

        # Hough Line Transform
        lines_detected = cv2.HoughLinesP(
            edges_for_lines,
            rho=1,
            theta=np.pi / 180,
            threshold=40, # Threshold from old horizontal detection
            minLineLength=50, # MinLineLength from old horizontal detection
            maxLineGap=20, # MaxLineGap from old horizontal detection
        )
        
        # Draw the detected lines onto the final_display_frame
        if lines_detected is not None:
            for line in lines_detected:
                x1, y1, x2, y2 = line[0]
                # You can filter by angle or length here if you only want to draw specific types of lines
                # For example, to draw horizontal lines (similar to old horizontal detection):
                # angle_rad = np.arctan2(y2 - y1, x2 - x1)
                # angle_deg = abs(np.degrees(angle_rad))
                # if angle_deg <= 20 or angle_deg >= (180 - 20): # horizontal_angle_threshold = 20
                #    cv2.line(final_display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red lines for horizontal
                # else:
                cv2.line(final_display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue lines for all lines (or other color)

        return final_display_frame # Return the frame with saturation mask, ROI overlay, and drawn lines

    # Martian Detection
    frame, action_flags["not_alone_timer_end"], martian_found = orb_detector.detect_martian_orb(
        frame, action_flags["not_alone_timer_end"]
    )
    if martian_found:
        robot_api.api.v2.movement.stop() # Corrected path if needed
        return frame 
    
    # Default case: if no path detected, no obstacle, no martian, and no procedure running.
    # Return the saturated frame (converted to BGR for consistent display).
    return cv2.cvtColor(saturated_frame_mask, cv2.COLOR_GRAY2BGR)
