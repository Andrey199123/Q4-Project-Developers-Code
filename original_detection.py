import copy
import math
import time

import cv2
import numpy as np


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


def compute_line(lines):
    used = [False] * len(lines)
    ans = []

    for i in range(len(lines)):
        if used[i] == False:
            used[i] == True

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
        if compute_slope(line) > 0.5 and compute_length(line) > compute_length(
            pos
        ):
            pos = line.copy()
        elif compute_slope(line) < -0.5 and compute_length(
            line
        ) > compute_length(neg):
            neg = line.copy()

    positive_slope = compute_slope(pos)
    negative_slope = compute_slope(neg)

    if (
        -negative_slope > positive_slope - 0.5
        and -negative_slope < positive_slope + 0.5
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
        cv2.line(frame, (pos[0], pos[1]), (pos[2], pos[3]), (0, 255, 0), 10)
        cv2.line(frame, (neg[0], neg[1]), (neg[2], neg[3]), (0, 255, 0), 10)
        return [temp + 1, pos[1], temp, pos[3]]
    else:
        return [0, 0, 0, 0]


def process_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 80, minLineLength=60, maxLineGap=20
    )

    n_lines = []
    centerline = []
    if lines is not None:
        for line in lines:
            print(compute_slope(line[0]))
    if lines is not None:
        n_lines = compute_line(lines)
        centerline = compute_center(n_lines, frame)

    if n_lines is not None:
        for line in n_lines:
            x1, y1, x2, y2 = line

            if compute_slope(line) < 0.1 and compute_slope(line) > -0.1:
               #cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 10)
                continue
            elif compute_length(line) > 200:
                #cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 10)
                continue

    if centerline and compute_slope(centerline) != 0:
        x1, y1, x2, y2 = centerline
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 10)

    return frame
