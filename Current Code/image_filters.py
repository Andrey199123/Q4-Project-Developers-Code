import cv2
import numpy as np

import cv2
import numpy as np

def is_significant_saturation_present(frame, saturation_threshold=0.4, min_percent=10):
    """
    Returns True if more than `min_percent` of the image has saturation greater than `saturation_threshold`.

    Args:
        frame: Input BGR image.
        saturation_threshold: Float in range [0, 1].
        min_percent: Minimum % of pixels required to return True.

    Returns:
        bool
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    total_pixels = frame.shape[0] * frame.shape[1]

    # Convert threshold from 0-1 to 0-255 scale
    threshold_value = int(saturation_threshold * 255)
    saturated_pixels = np.sum(saturation > threshold_value)

    percent_saturated = (saturated_pixels / total_pixels) * 100

    return percent_saturated >= min_percent

def apply_color_filter(hsv_image, lower_color_hsv, upper_color_hsv):
    """
    Applies a color mask to an HSV image.
    """
    mask = cv2.inRange(hsv_image, lower_color_hsv, upper_color_hsv)
    return mask
