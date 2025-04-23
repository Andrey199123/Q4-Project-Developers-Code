# ------- This is the image processing code from the robot. It was last updated April 21st, 2025 -------
import copy
import math
import time
import robot_api.api.v2.movement
import cv2
import numpy as np


def is_5_percent_blue(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    total_pixels = frame.shape[0] * frame.shape[1]
    blue_pixels = np.sum(mask > 0)
    blue_percentage = (blue_pixels / total_pixels) * 100
    return blue_percentage >= 5


def detect_turn(frame):
    def resize_frame(frame):
        height, width = frame.shape[:2]
        aspect_ratio = width / height
        new_height = 200
        new_width = int(new_height * aspect_ratio)
        return cv2.resize(frame, (new_width, new_height)), new_height, new_width, aspect_ratio

    def process_frame(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.25, beta=0)
        gray = cv2.medianBlur(gray, 5)
        return cv2.bilateralFilter(gray, 9, 75, 75)

    def sobel_edges(frame):
        sobel_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=7)
        sobel_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=7)
        edges = cv2.magnitude(sobel_x, sobel_y)
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, binary = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
        binary = cv2.dilate(binary, None, iterations=1)
        binary = cv2.erode(binary, None, iterations=1)
        binary = cv2.medianBlur(binary, 5)
        binary = cv2.bilateralFilter(binary, 9, 75, 75)
        lines = cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=100, minLineLength=25, maxLineGap=150)
        return lines

    def calculate_angle(line):
        x1, y1, x2, y2 = line
        angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi
        while angle < 0: angle += 180
        while angle > 180: angle -= 180
        return angle

    def is_left_turn(lines, width):
        left_count, right_count = 0, 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = calculate_angle((x1, y1, x2, y2))
            if angle < 45 or angle > 135:
                continue  # discard horizontal-ish lines
            if x1 < width // 2 and x2 < width // 2:
                left_count += 1
            elif x1 > width // 2 and x2 > width // 2:
                right_count += 1
        if left_count > right_count + 2:
            print('Left Turn')
        elif right_count > left_count + 2:
            print('Right Turn')
        else:
            print('Straight or Unknown')

    # Start of function logic
    resized, height, width, _ = resize_frame(frame)
    processed = process_frame(resized)
    lines = sobel_edges(processed)
    if lines is None or len(lines) < 2:
        return 'No lanes detected'
    return is_left_turn(lines, width)


def compute_slope(line):
    x1, y1, x2, y2 = line
    if x2 - x1 == 0:
        return 0
    return (y2 - y1) / (x2 - x1)


def compute_length(line):
    x1, y1, x2, y2 = line
    return np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)


def update(line, x1, y1, x2, y2, slope):
    x1 = min(x1, line[0])
    x2 = max(x2, line[2])
    if slope > 0:
        y2 = max(y2, line[3])
        y1 = min(y1, line[1])
    else:
        y2 = min(y2, line[3])
        y1 = max(y1, line[1])
    return x1, y1, x2, y2


def compute_line(lines, wack):
    used = [False] * len(lines)
    ans = []
    for i in range(len(lines)):
        if used[i] == False:
            used[i] = True
            x1, y1, x2, y2 = lines[i][0]
            base_slope = compute_slope(lines[i][0])
            if base_slope < 0.1 and base_slope > -0.1:
                for j in range(i, len(lines)):
                    temp = compute_slope(lines[j][0])
                    if (
                            used[j] == False
                            and temp < 0.1
                            and temp > -0.1
                            and lines[j][0][1] < y1 + 40
                            and lines[j][0][1] > y1 - 40
                    ):
                        x1, y1, x2, y2 = update(
                            lines[j][0], x1, y1, x2, y2, temp
                        )
                        used[j] = True
                ans.append([x1, int((y1 + y2) / 2), x2, int((y1 + y2) / 2)])
            else:
                for j in range(i, len(lines)):
                    temp = compute_slope(lines[j][0])
                    if (
                            used[j] == False
                            and temp < base_slope + 0.20
                            and temp > base_slope - 0.20
                    ):
                        x1, y1, x2, y2 = update(
                            lines[j][0], x1, y1, x2, y2, temp
                        )
                        used[j] = True
                ans.append([x1, y1, x2, y2])
    return ans


def compute_center(lines, frame):
    neg = [0, 0, 0, 0]
    pos = [0, 0, 0, 0]
    for line in lines:
        if compute_slope(line) > -2 and compute_length(line) > compute_length(
                pos
        ):
            pos = line.copy()
        elif compute_slope(line) < 2 and compute_length(
                line
        ) > compute_length(neg):
            neg = line.copy()
    positive_slope = compute_slope(pos)
    negative_slope = compute_slope(neg)
    if (
            positive_slope - 3.5 < abs(negative_slope) < positive_slope + 3.5
            and compute_length(pos) > 100
            and compute_length(neg) > 100
            and min(neg[0], neg[2]) < min(pos[0], pos[2])
    ):
        if neg[3] > pos[1]:
            if negative_slope != 0:
                neg[2] = int(neg[2] - (neg[3] - pos[1]) / negative_slope)
            neg[3] = pos[1]
        else:
            if positive_slope != 0:
                pos[0] = int(pos[0] + (neg[3] - pos[1]) / positive_slope)
            pos[1] = neg[3]
        if neg[1] < pos[3]:
            if negative_slope != 0:
                neg[0] = int(neg[0] - (neg[1] - pos[3]) / negative_slope)
            neg[1] = pos[3]
        else:
            if positive_slope != 0:
                pos[2] = int(pos[2] + (neg[1] - pos[3]) / positive_slope)
            pos[3] = neg[1]
        temp = int(
            (int((pos[2] + neg[0]) / 2) + int((pos[0] + neg[2]) / 2)) / 2
        )
        if (
                min(neg[0], neg[2]) < min(pos[0], pos[2])
        ):
            cv2.line(frame, (pos[0], pos[1]), (pos[2], pos[3]), (0, 255, 0), 10)
            cv2.line(frame, (neg[0], neg[1]), (neg[2], neg[3]), (0, 255, 0), 10)
            return [temp + 1, pos[1], temp, pos[3]]
        else:
            return [0, 0, 0, 0]
    else:
        return [0, 0, 0, 0]


def process_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = grey.shape[:2]
    vertices = np.array([[(0, height), (0, height / 6), (width, height / 6), (width, height)]], np.int32)
    mask = np.zeros_like(grey)
    cv2.fillPoly(mask, vertices, 255)
    gray = cv2.bitwise_and(grey, mask)
    lower_blue = np.array([110, 85, 85])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    gray = cv2.bitwise_and(gray, gray, mask=mask)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    # empty = np.zeros_like(edges)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 80, minLineLength=60, maxLineGap=20
    )
    n_lines = []
    centerline = []
    detect_turn(frame)
    if is_5_percent_blue(frame):
        print("Over 5%")
        # robot_api.api.v2.movement.stop()
    else:
        print("Less than 5%")
    if lines is not None:
        n_lines = compute_line(lines, False)
        centerline = compute_center(n_lines, frame)
    if n_lines is not None:
        for line in n_lines:
            x1, y1, x2, y2 = line
            # cv2.line(empty, (x1, y1), (x2,y2), (0,0,255), 10)
            if compute_slope(line) < 0.1 and compute_slope(line) > -0.1:
                # cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 10)
                continue
            elif compute_length(line) > 200:
                # cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 10)
                continue
    if centerline and compute_slope(centerline) != 0:
        x1, y1, x2, y2 = centerline
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return frame
