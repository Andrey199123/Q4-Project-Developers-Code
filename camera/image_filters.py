import cv2
import numpy as np

def is_significant_tape_present(frame):
    """hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    total_pixels = frame.shape[0] * frame.shape[1]

    # Define HSV ranges for each color
    color_ranges = {
        'blue':   ([100, 50, 50], [130, 255, 255]),
        'green':  ([70, 120, 100], [160, 190, 170]),
        'orange': ([10, 100, 100], [25, 255, 255]),
        'red1':   ([0, 70, 50], [10, 255, 255]),
        'red2':   ([170, 70, 50], [180, 255, 255])  # Red wraps around hue spectrum
    }

    # Helper function to calculate percentage of a masked color
    def get_color_percentage(lower, upper):
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        return (np.sum(mask > 0) / total_pixels) * 100

    # Check blue
    blue_percent = get_color_percentage(*color_ranges['blue'])
    if blue_percent >= 5:
        return True

    # Check green
    green_percent = get_color_percentage(*color_ranges['green'])
    if green_percent >= 2:
        return True

    # Check orange
    orange_percent = get_color_percentage(*color_ranges['orange'])
    if orange_percent >= 5:
        return True

    # Check red (combined from two ranges)
    red1_percent = get_color_percentage(*color_ranges['red1'])
    red2_percent = get_color_percentage(*color_ranges['red2'])
    if red1_percent + red2_percent >= 5:
        return True

    # No significant color found
    return False"""
    saturation_threshold = 0.4
    min_percent = 2
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    total_pixels = frame.shape[0] * frame.shape[1]

    # Convert threshold from 0-1 to 0-255 scale
    threshold_value = int(saturation_threshold * 255)
    saturated_pixels = np.sum(saturation > threshold_value)

    percent_saturated = (saturated_pixels / total_pixels) * 100
    print(percent_saturated)
    return percent_saturated >= min_percent

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

    return filtered_mask
def apply_color_filter(hsv_image, lower_color_hsv, upper_color_hsv):
    """
    Applies a color mask to an HSV image.
    """
    mask = cv2.inRange(hsv_image, lower_color_hsv, upper_color_hsv)
    return mask
