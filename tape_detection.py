import cv2
import numpy as np

def is_significant_saturation_present(frame, saturation_threshold=0.4, min_percent=10, min_region_size=600):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    total_pixels = frame.shape[0] * frame.shape[1]

    threshold_value = int(saturation_threshold * 255)
    binary_mask = (saturation > threshold_value).astype(np.uint8)

    saturated_pixels = np.sum(binary_mask)
    percent_saturated = (saturated_pixels / total_pixels) * 100

    # Connected components analysis to remove small noisy blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # Filter out small regions
    filtered_mask = np.zeros_like(binary_mask)
    for i in range(1, num_labels):  # skip background label 0
        if stats[i, cv2.CC_STAT_AREA] >= min_region_size:
            filtered_mask[labels == i] = 255

    return percent_saturated >= min_percent, filtered_mask

def main():
    frame = cv2.imread("") # Put the file path here

    if frame is None:
        print("Error: Could not load image.")
        return

    while True:
        _, saturation_mask = is_significant_saturation_present(frame)
        cv2.imshow('Original Image', frame)
        cv2.imshow('Filtered Saturation Mask', saturation_mask)

        # Press 'q' to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
