import numpy as np
import cv2

# --- Global state variables for smoothing and detection (keep if still used for other things, but less critical here) ---
# previous_lines_state is no longer relevant as we're not using lines.
# previous_mid_x is no longer relevant for line following.
previous_mid_x = None # Keep for now in case other modules use it, but it won't be updated by this file's core logic.
alpha = 0.9 # Smoothing factor (less relevant for pure region detection)

def saturation_mask_dark(frame, saturation_threshold=0.7, value_threshold=0.8, min_percent=10, min_region_size=600):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    total_pixels = frame.shape[0] * frame.shape[1]

    saturation_mask = saturation < saturation_threshold * 255
    value_mask = value > value_threshold * 255
    binary_mask = np.logical_and(saturation_mask, value_mask).astype(np.uint8) * 255

    # Connected components analysis to remove small noisy blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # Filter out small regions
    filtered_mask = np.zeros_like(binary_mask)
    for i in range(1, num_labels):  # skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_region_size:
            filtered_mask[labels == i] = 255

    return filtered_mask


def saturation_mask(frame2, saturation_threshold=0.2, min_percent=10, min_region_size=600):
    frame = cv2.convertScaleAbs(frame2, alpha=1, beta=0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    total_pixels = frame.shape[0] * frame.shape[1]

    threshold_value = int(saturation_threshold * 255)
    binary_mask = (saturation > threshold_value).astype(np.uint8) * 255

    # Connected components analysis to remove small noisy blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # Filter out small regions
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

def ROI(image, frame_width, frame_height):
    """
    Creates a binary mask for the Region of Interest (ROI) at a specified height
    and covering the left and right thirds of the frame horizontally.
    Returns the image masked by the ROI and the binary ROI mask itself.
    """
    roi_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    
    # X-COORDINATES (keeping them as left and right thirds)
    left_x_start = 0
    left_x_end = int(frame_width * 0.33)
    right_x_start = int(frame_width * 0.66)
    right_x_end = frame_width
    
    # --- ADJUSTED Y-COORDINATES (placing ROI higher up) ---
    # Define the height (thickness) of the ROI band
    roi_height_pixels = int(frame_height * 0.1) # ROI band thickness, adjust as needed (e.g., 0.05 for thinner, 0.15 for thicker)
    
    # Define how high from the top of the frame the ROI should start.
    # A smaller percentage means higher up (closer to the top of the frame).
    # For example:
    # 0.2 means the top of the ROI starts at 20% down from the top.
    # 0.4 means the top of the ROI starts at 40% down from the top (higher than middle, assuming default 0.1 height).
    # You will need to fine-tune this value.
    percentage_from_top = 0.3 # <--- ADJUST THIS VALUE TO MOVE ROI UP/DOWN

    height_start = int(frame_height * percentage_from_top)
    height_end = height_start + roi_height_pixels
    
    # Ensure coordinates are within frame bounds
    height_start = max(0, height_start)
    height_end = min(frame_height, height_end)
    
    # Apply the ROI mask to the two regions
    roi_mask[height_start:height_end, left_x_start:left_x_end] = 255
    roi_mask[height_start:height_end, right_x_start:right_x_end] = 255
    
    # Apply the ROI mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=roi_mask)
    return masked_image, roi_mask


def determine_roi_presence(mask, frame_width, frame_height):
    """
    Determines if significant saturated pixels ("white") exist in the
    left and right sub-regions of the provided mask, based on the ROI definition.
    
    Args:
        mask (numpy.ndarray): The 1-channel binary mask (e.g., from saturation_mask)
                              that has already been constrained to the main ROI.
        frame_width (int): The width of the original frame.
        frame_height (int): The height of the original frame.

    Returns:
        tuple: A tuple containing (left_detected, right_detected),
               where each is a boolean indicating presence of significant white pixels.
    """
    # These x-coordinates define the left and right thirds for checking presence.
    # They match the horizontal definitions in the ROI function.
    left_x_end_check = int(frame_width * 0.33)
    right_x_start_check = int(frame_width * 0.66)
    
    # --- Y-COORDINATES: These MUST match the vertical placement logic in the ROI function ---
    # Define the height (thickness) of the ROI band.
    # This value MUST be the same as 'roi_height_pixels' used in the ROI function.
    roi_height_pixels = int(frame_height * 0.1) 
    
    # Define how high from the top of the frame the ROI starts.
    # This value MUST be the same as 'percentage_from_top' used in the ROI function.
    percentage_from_top = 0.3 # <--- ENSURE THIS MATCHES THE VALUE IN THE ROI FUNCTION

    height_start_check = int(frame_height * percentage_from_top)
    height_end_check = height_start_check + roi_height_pixels
    
    # Ensure coordinates are within frame bounds
    height_start_check = max(0, height_start_check)
    height_end_check = min(frame_height, height_end_check)

    # Slice the input 'mask' (which is already limited to the overall ROI)
    # to get the specific left and right sub-regions.
    left_roi_section = mask[height_start_check:height_end_check, 0:left_x_end_check]
    right_roi_section = mask[height_start_check:height_end_check, right_x_start_check:frame_width]

    # Check for non-zero (white/saturated) pixels in each section.
    # The threshold (100) can be adjusted based on testing for sensitivity.
    left_detected = np.count_nonzero(left_roi_section) > 100
    right_detected = np.count_nonzero(right_roi_section) > 100
    
    return left_detected, right_detected

def determine_direction(left_detected, right_detected):
    """
    Determines the robot's turn direction based on ROI presence.
    """
    if left_detected and right_detected:
        return "Middle"
    elif left_detected:
        return "Right"
    elif right_detected:
        return "Left"
    else:
        return "Right"
def saturation_mask(frame2, saturation_threshold=0.2, min_percent=10, min_region_size=600):
    frame = cv2.convertScaleAbs(frame2, alpha=1, beta=0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    total_pixels = frame.shape[0] * frame.shape[1]

    threshold_value = int(saturation_threshold * 255)
    binary_mask = (saturation > threshold_value).astype(np.uint8) * 255

    # Connected components analysis to remove small noisy blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # Filter out small regions
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


# --- Core Direction Determination Function (Simplified) ---
def get_direction_from_roi_saturation(frame, line_state):
    """
    Determines direction based purely on saturation mask within defined ROIs.
    This replaces the complex line detection logic.
    """
    frame_height, frame_width = frame.shape[:2]

    # 1. Apply ROI to the frame (original frame, not saturated yet)
    roi_frame, _ = ROI(frame, frame_width, frame_height) # We only need roi_frame here for next step

    # 2. Get saturation mask from the ROI'd frame
    saturation_mask_result = saturation_mask(roi_frame)
    
    # 3. Determine presence of saturated pixels in left/right ROI sections
    left_detected, right_detected = determine_roi_presence(saturation_mask_result, frame_width, frame_height)
    
    # 4. Determine direction based on ROI presence
    current_direction = determine_direction(left_detected, right_detected)
    
    # Update line_state (keep consistent with previous usage)
    line_state["vertical_detected"] = left_detected or right_detected # Or set based on your interpretation
    line_state["horizontal_detected"] = False # No horizontal lines are detected by this method
    line_state["direction"] = current_direction
    
    # Return the processed saturation mask and the updated state.
    # We return the saturation_mask_result as it represents the "white" pixels.
    return saturation_mask_result, line_state
