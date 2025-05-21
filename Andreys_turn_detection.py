import cv2
import numpy as np
import math
from skimage.morphology import skeletonize

def detect_lines_in_mask(mask, min_line_length=50, max_line_gap=10, angle_range=(0, 180)):
    # Edge detection
    edges = cv2.Canny(mask, 50, 150)

    # Hough Line Transform to detect straight line segments
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=30,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    # Optional filtering by angle (e.g. ignore near-horizontal if you only want vertical segments)
    filtered_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            if angle_range[0] <= angle <= angle_range[1]:
                filtered_lines.append((x1, y1, x2, y2))
    return filtered_lines

def is_significant_saturation_present(frame, saturation_threshold=0.4, min_percent=10, min_region_size=600):
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

    percent_saturated = (np.sum(filtered_mask > 0) / total_pixels) * 100
    return percent_saturated >= min_percent, filtered_mask

def apply_perspective_transform(frame):
    h, w = frame.shape[:2]
    pts1 = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
    pts2 = pts1.copy()  # identity transform
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(frame, M, (w, h))

def main():
    frame = cv2.imread("/imgpath")
    if frame is None:
        print("Error: Could not load image.")
        return

    transformed = apply_perspective_transform(frame)

    while True:
        _, mask = is_significant_saturation_present(transformed)

        # fill small holes so skeleton is continuous
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # skeletonize to 1-px centerlines
        skel_bool = skeletonize(closed > 0)
        skeleton = (skel_bool.astype(np.uint8) * 255)

        # detect multiple segments on the skeleton
        lines = detect_lines_in_mask(skeleton,
                                     min_line_length=50,
                                     max_line_gap=10,
                                     angle_range=(0, 180))

        # draw all detected segments
        output = transformed.copy()
        for x1, y1, x2, y2 in lines:
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 50)

        # display
        cv2.imshow('Filtered Mask', mask)
        cv2.imshow('Skeleton', skeleton)
        cv2.imshow('Detected Path Segments', output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
