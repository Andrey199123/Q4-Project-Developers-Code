# robot_api/camera/roi_utils.py
import numpy as np
import cv2

def apply_main_roi_mask(image, frame_width, frame_height):
    """
    Applies a predefined ROI mask to the image.
    The mask covers the full width but focuses on the upper 5/6ths of the height.
    """
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    vertices = np.array(
        [[(0, frame_height), (0, int(frame_height / 6)), (frame_width, int(frame_height / 6)), (frame_width, frame_height)]],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image, mask


def apply_three_lane_roi_mask(image, frame_width, frame_height):
    """
    Creates and applies an ROI mask dividing the frame into three vertical lanes.
    """
    roi_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

    left_width = int(frame_width * 0.33)
    middle_start = int(frame_width * 0.33)
    middle_end = int(frame_width * 0.66)
    right_end = frame_width

    roi_mask[:, 0:left_width] = 255
    roi_mask[:, middle_start:middle_end] = 255
    roi_mask[:, middle_end:right_end] = 255 # Corrected to use middle_end to right_end

    if len(image.shape) == 3 and image.shape[2] == 3 : # Check if it's a color image
        masked_image = cv2.bitwise_and(image, image, mask=roi_mask)
    elif len(image.shape) == 2: # Grayscale image
        masked_image = cv2.bitwise_and(image, image, mask=roi_mask)
    else:
        print("Unsupported image format for ROI masking.")
        masked_image = image # Return original if format is unexpected

    return masked_image, roi_mask


def determine_line_roi_position(x_coordinate, frame_width):
    """
    Determines which ROI (Left, Middle, Right) an x-coordinate belongs to
    based on a slightly different division for line guidance.
    """
    left_threshold = int(frame_width * 0.40) 
    right_threshold = int(frame_width * 0.60)

    if x_coordinate < left_threshold:
        return "Left"
    elif x_coordinate > right_threshold:
        return "Right"
    else:
        return "Middle"

