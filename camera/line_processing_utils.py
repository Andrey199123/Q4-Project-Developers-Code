import numpy as np
import cv2

# --- Geometric Helper Functions ---
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def smooth_value(new_value, previous_value, alpha=0.9):
    if previous_value is None:
        return new_value
    return int(alpha * previous_value + (1 - alpha) * new_value)

def compute_slope(line_segment):
    x1, y1, x2, y2 = line_segment
    if x2 - x1 == 0:  # Vertical line
        return float('inf') if y2 > y1 else float('-inf') # Avoid division by zero, return signed infinity
    return (y2 - y1) / (x2 - x1)

def compute_length(line_segment):
    x1, y1, x2, y2 = line_segment
    return np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

# --- Line Filtering and Grouping ---
def filter_lines_by_angle(lines, angle_threshold=20, previous_lines_state=None):
    """
    Filters lines to keep near-vertical and near-horizontal lines.
    Uses previous lines if no new lines are found.
    """
    if previous_lines_state is None:
        previous_lines_state = []

    if lines is None:
        return previous_lines_state, previous_lines_state # Return current state and previous lines

    current_filtered_lines = []
    for line_arr in lines:
        line = line_arr[0] # HoughLinesP returns lines inside an array
        x1, y1, x2, y2 = line
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

        # Vertical-ish lines
        if 90 - angle_threshold <= angle <= 90 + angle_threshold:
            # Standardize direction (e.g., top to bottom)
            if y2 < y1: # If y2 is above y1, swap points
                x1, y1, x2, y2 = x2, y2, x1, y1
            current_filtered_lines.append([x1, y1, x2, y2])
        # Horizontal-ish lines
        elif angle <= angle_threshold or angle >= (180 - angle_threshold):
            # Standardize direction (e.g., left to right)
            if x2 < x1: # If x2 is to the left of x1, swap points
                x1, y1, x2, y2 = x2, y2, x1, y1
            current_filtered_lines.append([x1, y1, x2, y2])

    if current_filtered_lines:
        return current_filtered_lines, current_filtered_lines # Return new lines and update state
    else:
        return previous_lines_state, previous_lines_state # Return old lines if no new ones passed filter

def get_dominant_vertical_lines(lines, frame_width):
    """
    From a list of predominantly vertical lines, tries to find the two most dominant
    lines representing lane boundaries.
    """
    if not lines or len(lines) < 1: # Need at least one to try and find a pair or center
        return []

    # Sort lines by their average x-coordinate to help distinguish left/right
    lines.sort(key=lambda line: (line[0] + line[2]) / 2)

    # Separate lines into left and right based on frame center
    mid_frame_x = frame_width // 2
    left_lines = [line for line in lines if (line[0] + line[2]) / 2 < mid_frame_x]
    right_lines = [line for line in lines if (line[0] + line[2]) / 2 >= mid_frame_x]

    main_lines = []
    if left_lines:
        # Choose the rightmost of the left lines (closest to center from left)
        left_lines.sort(key=lambda line: (line[0] + line[2]) / 2, reverse=True)
        main_lines.append(left_lines[0])
        
    if right_lines:
        # Choose the leftmost of the right lines (closest to center from right)
        right_lines.sort(key=lambda line: (line[0] + line[2]) / 2)
        main_lines.append(right_lines[0])

    if len(main_lines) == 2:
         # Ensure left line is on the left of right line
        if (main_lines[0][0] + main_lines[0][2]) / 2 > (main_lines[1][0] + main_lines[1][2]) / 2:
            main_lines.reverse() # Swap them
        return main_lines
    elif len(main_lines) == 1: # Only one dominant line found
        return main_lines
    
    return []


def draw_lines_and_interpret_lanes(frame, lines, line_state, roi_determine_func):
    """
    Draws filtered lines and determines lane center and direction.
    Modifies line_state dictionary with new values.
    roi_determine_func is determine_line_roi_position from roi_utils.
    """
    line_image = np.zeros_like(frame)
    current_direction = line_state.get("direction", 0) # Default to 0 or last known
    vertical_detected_flag = False

    if not lines:
        line_state["vertical_detected"] = False
        return line_image, line_state # Return empty line_image and current state

    # Draw all filtered lines
    for x1, y1, x2, y2 in lines:
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 3) # Blue for all filtered lines

    # Separate lines by orientation
    vertical_lines = []
    horizontal_lines = [] # For potential T-junction or stop line detection
    for line in lines:
        angle = abs(np.degrees(np.arctan2(line[3] - line[1], line[2] - line[0])))
        if 70 < angle < 110: # More strictly vertical
            vertical_lines.append(line)
        elif angle < 20 or angle > 160: # More strictly horizontal
            horizontal_lines.append(line)

    # --- Vertical Lane Following Logic ---
    if vertical_lines:
        main_vertical_lanes = get_dominant_vertical_lines(vertical_lines, frame.shape[1])
        
        if len(main_vertical_lanes) == 2: # Two lane lines found
            left_x_avg = (main_vertical_lanes[0][0] + main_vertical_lanes[0][2]) / 2
            right_x_avg = (main_vertical_lanes[1][0] + main_vertical_lanes[1][2]) / 2
            
            mid_x = int((left_x_avg + right_x_avg) / 2)
            smoothed_mid_x = smooth_value(mid_x, line_state.get("previous_mid_x"))
            line_state["previous_mid_x"] = smoothed_mid_x
            current_direction = roi_determine_func(smoothed_mid_x, frame.shape[1])
            vertical_detected_flag = True
            
            cv2.line(line_image, (smoothed_mid_x, 0), (smoothed_mid_x, frame.shape[0]), (0, 0, 255), 3) # Red center line
            for x1,y1,x2,y2 in main_vertical_lanes:
                 cv2.line(line_image, (x1,y1), (x2,y2), (0,255,0), 5)


        elif len(main_vertical_lanes) == 1:
            line_x_avg = (main_vertical_lanes[0][0] + main_vertical_lanes[0][2]) / 2
            frame_width = frame.shape[1]
            edge_threshold = frame_width * 0.33  # ~15% from the edge
            
            if line_x_avg < edge_threshold:
                current_direction = "Right"  # Left border only → turn right
            elif line_x_avg > frame_width - edge_threshold:
                current_direction = "Left"  # Right border only → turn left
            else:
                current_direction = "Middle"  # Line near center, maybe just drift
            
            smoothed_mid_x = smooth_value(int(line_x_avg), line_state.get("previous_mid_x"))
            line_state["previous_mid_x"] = smoothed_mid_x
            vertical_detected_flag = True
            cv2.line(line_image, (smoothed_mid_x, 0), (smoothed_mid_x, frame.shape[0]), (0, 165, 255), 3)
            for x1, y1, x2, y2 in main_vertical_lanes:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), 5)
    line_state["vertical_detected"] = vertical_detected_flag
    line_state["direction"] = current_direction
    
    if len(horizontal_lines) >=1: # If at least one strong horizontal line
        line_state["horizontal_detected"] = True
    else:
        line_state["horizontal_detected"] = False
        
    return line_image, line_state


def detect_significant_horizontal_lines(frame, roi_processing_func, color_filter_func):
    """
    Detects if significant horizontal lines exist within specific color ranges and ROIs.
    This is a refactor of the original `detect_horizontal_line`.

    Args:
        frame (numpy.ndarray): The input video frame (BGR format).
        roi_processing_func (function): A function to apply ROI, e.g., from roi_utils.
                                        Expected signature: func(image, width, height) -> masked_image, roi_mask
        color_filter_func (function): A function to filter by color.
                                      Expected signature: func(image_hsv) -> color_mask

    Returns:
        bool: True if at least two distinct horizontal lines are detected.
    """
    horizontal_angle_threshold = 20  # Degrees for horizontal
    min_horizontal_lines_required = 1 # Changed from 2 to 1 to be less strict for T-junctions

    if frame is None:
        print("Error: Input frame is None for horizontal line detection.")
        return False

    frame_height, frame_width = frame.shape[:2]

    roi_frame, _ = roi_processing_func(frame, frame_width, frame_height)
    
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    
    lower_color = np.array([90, 80, 40]) # Broadened slightly
    upper_color = np.array([150, 255, 255])
    color_mask = cv2.inRange(hsv, lower_color, upper_color)
    
    masked_frame_roi = cv2.bitwise_and(roi_frame, roi_frame, mask=color_mask)

    gray = cv2.cvtColor(masked_frame_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=40,
        minLineLength=frame_width // 10, # Min length relative to frame width
        maxLineGap=frame_width // 20      # Max gap relative to frame width
    )

    horizontal_line_count = 0
    if lines is not None:
        for line_arr in lines:
            line = line_arr[0]
            x1, y1, x2, y2 = line
            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            angle_deg = abs(np.degrees(angle_rad))

            if angle_deg <= horizontal_angle_threshold or \
               angle_deg >= (180 - horizontal_angle_threshold):
                horizontal_line_count += 1
    
    return horizontal_line_count >= min_horizontal_lines_required
