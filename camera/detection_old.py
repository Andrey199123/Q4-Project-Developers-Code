import copy
import math
import robot_api.api.v2.movement
import time
import threading
import cv2
import numpy as np
import os.path
from gpiozero import DistanceSensor

ultrasonic = DistanceSensor(trigger=27, echo=17)

MARTIAN_PATH = "/home/insomnia/Documents/Andreytesting/robot_api/camera/martian.png"

previous_lines = []
previous_mid_x = None
alpha = 0.9
Vertical_Detected = False
Horizontal_Detected = False
direction = 0
not_alone = 0
martian = cv2.imread(MARTIAN_PATH, 0)
print(os.path.exists(MARTIAN_PATH))
def ORB(frame):
    """Detect and display the Martian on each frame.

    Keyword arguments:
       frame -- An image captured from a live video stream,
                Mat | ndarray[Any, dtype] | ndarray = (cap. read())[1]
    Returns:
       frame -- An image with either no change or a text box displaying
                'WE ARE NOT ALONE!!'
    """

    global martian
    global not_alone
# Resize to dimensions of the martian template for ORB
    frame_blur = cv2.blur(frame,(19,19)) # Reduce noise for less inaccurate matches
    frame_gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()

    keypoints1, descriptors1 = orb.detectAndCompute(martian, None)
    keypoints2, descriptors2 = orb.detectAndCompute(frame_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    try:
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        num_matches = len(matches)
        print(num_matches)

        if num_matches > 10 or not_alone - time.time() > 0:
            if num_matches > 10:
                print("WE ARE NOT ALONE")
                not_alone = time.time() + 5
            robot_api.api.v2.movement.stop()
            show_not_alone(frame)
        else:
            print("nothing")
    except Exception as e:
        print("Error in matching", e)
    return frame

def show_not_alone(frame):
   """Put a text box on the frame.

   Keyword arguments:
       frame -- An image captured from a live video stream,
               Mat | ndarray[Any, dtype] | ndarray = (cap. read())[1]

   Returns:
      frame -- An image with a red text box and white lettering displaying
               'WE ARE NOT ALONE!!' in the top left corner.
   """

   font = cv2.FONT_HERSHEY_SIMPLEX
   font_scale = 1
   color = (255, 255, 255)  # White color
   thickness = 2
   position = (10, 50)  # Coordinates of the bottom-left corner of the text string in the image

   text = "WE ARE NOT ALONE!!"

   text_size = cv2.getTextSize(text, font, font_scale, thickness)[0] # Get text size to create a background rectangle
   text_width, text_height = text_size

   rectangle_color = (0, 0, 255)  # Red color
   rectangle_position1 = position  # Top-left corner of the rectangle is the same as the text position
   rectangle_position2 = (position[0] + text_width, position[1] - text_height - 10)  # Bottom-right corner

   cv2.rectangle(frame, rectangle_position1, rectangle_position2, rectangle_color, -1)

   cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)


def ROI(image, frame_width, frame_height):
    # Create a blank ROI mask the same size as the image
    roi_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

    # Define ROI regions (left and middle)
    left_width = int(frame_width * 0.33)  # Left region: 33% of frame width
    middle_start = int(frame_width * 0.33)
    middle_end = int(
        frame_width * 0.66
    )  # Middle region: 33%-66% of frame width
    right_width = int(frame_width)  # Right region: 33% of frame width
    # Fill the ROI mask for left region
    roi_mask[:, 0:left_width] = 255
    roi_mask[:, middle_end:right_width] = 255
    # Fill the ROI mask for middle region
    roi_mask[:, middle_start:middle_end] = 255
    """if len(image.shape) == 3:
        # Convert the image to grayscale if it's not already
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)"""
    # Apply the ROI mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=roi_mask)
    # cv2.imshow('ROI Mask', masked_image)

    return masked_image, roi_mask


def determine_roi(x, frame_width):
    """
    Determines which ROI the x-coordinate belongs to.
    """
    left_width = int(frame_width * 0.1)
    middle_start = int(frame_width * 0.33)
    middle_end = int(frame_width * 0.5)

    if x < left_width:
        return "Left"
    elif middle_start <= x < middle_end:
        return "Middle"
    else:
        print(x, frame_width)
        return "Right"


def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def smooth_value(new_value, previous_value, alpha=0.9):
    if previous_value is None:
        return new_value
    return int(alpha * previous_value + (1 - alpha) * new_value)


def filter_lines(lines, angle_threshold=20):
    global previous_lines

    if lines is None:
        if previous_lines:
            return previous_lines
        return []

    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

        if 90 - angle_threshold <= angle <= 90 + angle_threshold:
            if y2 > y1:
                x1, y1, x2, y2 = x2, y2, x1, y1
            filtered_lines.append([x1, y1, x2, y2])

        if angle <= angle_threshold or angle >= 180 - angle_threshold:
            if x2 < x1:
                x1, y1, x2, y2 = x2, y2, x1, y1
            filtered_lines.append([x1, y1, x2, y2])

    if filtered_lines:
        previous_lines = filtered_lines
    elif previous_lines:
        return previous_lines

    return filtered_lines


def get_two_main_lines(lines, frame_width):
    if len(lines) < 2:
        return []

    lines.sort(key=lambda line: line[0])

    left_lines = []
    right_lines = []
    mid_x = frame_width // 2

    for line in lines:
        x1 = line[0]
        if x1 < mid_x:
            left_lines.append(line)
        else:
            right_lines.append(line)

    if left_lines and right_lines:
        left_line = max(left_lines, key=lambda line: line[0])
        right_line = min(right_lines, key=lambda line: line[0])
        return [left_line, right_line]
    return []


def draw_lines_and_center(frame, lines):
    global previous_mid_x, Vertical_Detected, Horizontal_Detected, direction
    line_image = np.zeros_like(frame)
    line_detected = False
    for x1, y1, x2, y2 in lines:
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # Filter lines by angle
    vertical_lines = [
        line
        for line in lines
        if abs(np.degrees(np.arctan2(line[3] - line[1], line[2] - line[0])))
        > 70
    ]
    horizontal_lines = [
        line
        for line in lines
        if abs(np.degrees(np.arctan2(line[3] - line[1], line[2] - line[0])))
        < 20
    ]

    # Handle vertical lines
    main_vertical_lines = get_two_main_lines(vertical_lines, frame.shape[1])
    if len(main_vertical_lines) == 2:
        left_x = main_vertical_lines[0][0]
        right_x = main_vertical_lines[1][0]
        mid_x = int((left_x + right_x) / 2)
        mid_x = smooth_value(mid_x, previous_mid_x)
        previous_mid_x = mid_x
        direction = determine_roi(mid_x, frame.shape[1])
        Vertical_Detected = True
        # print(f"Vertical Detected: {Vertical_Detected}, Mid X: {mid_x}")
        height = frame.shape[0]
        # cv2.line(line_image, (mid_x, 0), (mid_x, height), (0, 0, 255), 3)
    else:
        Vertical_Detected = False

    # Handle horizontal lines
    if len(horizontal_lines) >= 2:
        horizontal_lines.sort(key=lambda line: line[1])
        top_line = horizontal_lines[0]
        bottom_line = horizontal_lines[-1]
        line_detected = True
        direction = determine_roi(top_line[0], frame.shape[1])
        # print(f"Horizontal Detected: {line_detected}")
        mid_y = int((top_line[1] + bottom_line[1]) / 2)
        width = frame.shape[1]
        # cv2.line(line_image, (0, mid_y), (width, mid_y), (0, 0, 255), 3)
    else:
        line_detected = False

    return line_detected, direction


def is_5_percent_blue(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    total_pixels = frame.shape[0] * frame.shape[1]
    blue_pixels = np.sum(mask > 0)
    blue_percentage = (blue_pixels / total_pixels) * 100
    return blue_percentage >= 5


def detect_horizontal_line(frame):
    """
    Processes a single video frame to detect if significant horizontal lines
    exist within specific color ranges and Regions of Interest (ROI).

    Args:
        frame (numpy.ndarray): The input video frame (BGR format).

    Returns:
        bool: True if at least two distinct horizontal lines are detected based
              on angle, False otherwise.
    """
    # --- Constants ---
    horizontal_angle_threshold = 20  # Degrees within which a line is considered horizontal (near 0 or 180)
    min_horizontal_lines_required = (
        2  # Minimum number of horizontal lines to detect
    )

    # --- Get Frame Dimensions ---
    if frame is None:
        print("Error: Input frame is None.")
        return False  # Cannot detect lines in a None frame

    frame_height, frame_width = frame.shape[:2]

    # --- Nested Helper Function for ROI ---
    # Kept because the detection logic relies on focusing on specific image parts
    def _ROI(image, width, height):
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        left_width = int(width * 0.33)
        middle_start = int(width * 0.33)
        middle_end = int(width * 0.66)
        # Fill the ROI mask for left region
        roi_mask[:, 0:left_width] = 255
        # Fill the ROI mask for middle region
        roi_mask[:, middle_start:middle_end] = 255
        # Apply the ROI mask to the image
        masked_image = cv2.bitwise_and(image, image, mask=roi_mask)
        # We don't need to return the mask itself for the final boolean result
        return masked_image

    roi_frame = _ROI(frame, frame_width, frame_height)
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    lower_color = np.array([90, 100, 50])
    upper_color = np.array([150, 255, 255])
    color_mask = cv2.inRange(hsv, lower_color, upper_color)
    masked_frame_roi = cv2.bitwise_and(roi_frame, roi_frame, mask=color_mask)

    gray = cv2.cvtColor(masked_frame_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=50,
        maxLineGap=20,
    )

    # 5. Count Horizontal Lines based on Angle
    horizontal_line_count = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate angle in degrees
            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            angle_deg = abs(np.degrees(angle_rad))

            # Check if the angle is within the horizontal thresholds (close to 0 or 180)
            if angle_deg <= horizontal_angle_threshold or angle_deg >= (
                180 - horizontal_angle_threshold
            ):
                horizontal_line_count += 1

    # 6. Determine Detection Status based on Count
    line_detected = horizontal_line_count >= min_horizontal_lines_required
    return line_detected


def compute_slope(line):
    x1, y1, x2, y2 = line

    if x2 - x1 == 0:
        return 0

    return (y2 - y1) / (x2 - x1)
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

def compute_center(line, frame):
    neg = [0, 0, 0, 0]
    pos = [0, 0, 0, 0]
    if compute_slope(line) > -2 and compute_length(line) > compute_length(
            pos
        ):
        pos = line.copy()
    elif compute_slope(line) < 2 and compute_length(line) > compute_length(
            neg
        ):
       neg = line.copy()

    positive_slope = compute_slope(pos)
    negative_slope = compute_slope(neg)
    if (abs(negative_slope) > positive_slope - 3.5
        and abs(negative_slope) < positive_slope + 3.5
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
        temp = int(int((pos[2] + neg[0]) / 2) + int((pos[0] + neg[2]) / 2)) / 2
        if min(neg[0], neg[2]) < min(pos[0], pos[2]):
            cv2.line(frame, (pos[0], pos[1]), (pos[2], pos[3]), (0, 255, 0), 10)
            cv2.line(frame, (neg[0], neg[1]), (neg[2], neg[3]), (0, 255, 0), 10)
            return [temp + 1, pos[1], temp, pos[3]]
        else:
            return [0, 0, 0, 0]
    else:
        return [0, 0, 0, 0]


CURRENTLY_RUNNING_PROCEDURE = False
CURRENTLY_RUNNING_PROCEDURE2 = False
proc2 = False
turned_left = False
turned_right = False
object_detected = False
def process_frame(frame):
    global CURRENTLY_RUNNING_PROCEDURE
    global CURRENTLY_RUNNING_PROCEDURE2
    global not_alone
    global object_detected

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = grey.shape[:2]
    vertices = np.array(
        [[(0, height), (0, height / 6), (width, height / 6), (width, height)]],
        np.int32,
    )
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

    frame_width = width
    frame_height = height
    roi_frame, roi_mask = ROI(frame, frame_width, frame_height)
    roi_visualization = frame.copy()
    colored_mask = np.zeros_like(frame)
    colored_mask[roi_mask == 255] = [0, 255, 0]  # Green color
    roi_visualization = cv2.addWeighted(
        roi_visualization, 1, colored_mask, 0.3, 0
    )
    # cv2.imshow('ROI Visualization', roi_visualization)
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    lower_color = np.array([90, 100, 50])  # Broadened lower bound (H, S, V)
    upper_color = np.array([150, 255, 255])  # Broadened upper bound (H, S, V)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Apply the mask to the original frame (only show tape-colored areas)
    masked_frame = cv2.bitwise_and(roi_frame, roi_frame, mask=mask)

    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)

    # Detect lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100, #Changed from 50
        minLineLength=80,
        maxLineGap=10,
    )
    frame = ORB(frame)

    if(ultrasonic.wait_for_in_range):
       print(ultrasonic.distance * 100) 
       if ultrasonic.distance * 100 < 20:
          object_detected = True
       else:
          object_detected = False
    if object_detected:
        def back():
            global CURRENTLY_RUNNING_PROCEDURE
            if not CURRENTLY_RUNNING_PROCEDURE:
                CURRENTLY_RUNNING_PROCEDURE = True
                robot_api.api.v2.movement.stop()
                robot_api.api.v2.movement.backward()
                time.sleep(3)
                robot_api.api.v2.movement.stop()
                CURRENTLY_RUNNING_PROCEDURE = False
                object_detected = False
        if not CURRENTLY_RUNNING_PROCEDURE:
           threading.Thread(target=back).start()
    # Filter and draw lines
    filtered_lines = filter_lines(lines)
    if filtered_lines:
        global direction
        line_image, direction = draw_lines_and_center(frame, filtered_lines)
        frame = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    # Show the result
    # cv2.imshow('Line Detection', frame)
    print(direction)
    global turned_left
    global turned_right
    if direction == "Left":
        def left():
            global CURRENTLY_RUNNING_PROCEDURE
            if not CURRENTLY_RUNNING_PROCEDURE:
                CURRENTLY_RUNNING_PROCEDURE = True
                robot_api.api.v2.movement.stop()
                robot_api.api.v2.movement.forward()
                time.sleep(6.5)
                robot_api.api.v2.movement.stop()
                robot_api.api.v2.movement.left()
                time.sleep(1.4)
                robot_api.api.v2.movement.stop()
                robot_api.api.v2.movement.forward()
                CURRENTLY_RUNNING_PROCEDURE = False

        if not CURRENTLY_RUNNING_PROCEDURE:
            if not turned_left:
                threading.Thread(target=left).start()
                turned_left = True
    elif direction == "Right":

        def right():
            global CURRENTLY_RUNNING_PROCEDURE
            if not CURRENTLY_RUNNING_PROCEDURE:
                CURRENTLY_RUNNING_PROCEDURE = True
                robot_api.api.v2.movement.stop()
                robot_api.api.v2.movement.forward()
                time.sleep(5.5)
                robot_api.api.v2.movement.stop()
                robot_api.api.v2.movement.right()
                time.sleep(1.47)
                robot_api.api.v2.movement.stop()
                robot_api.api.v2.movement.forward()
                CURRENTLY_RUNNING_PROCEDURE = False

        if not CURRENTLY_RUNNING_PROCEDURE:
            if not turned_right:
                threading.Thread(target=right).start()
                turned_right = True
    if detect_horizontal_line(frame):
        pass
        # print("HORIZONTAL LINE DETECTED")
        """def turn_right():
            global CURRENTLY_RUNNING_PROCEDURE
            if not CURRENTLY_RUNNING_PROCEDURE:
                CURRENTLY_RUNNING_PROCEDURE = True
                robot_api.api.v2.movement.stop()
                robot_api.api.v2.movement.forward()
                time.sleep(4)
                robot_api.api.v2.movement.stop()
                robot_api.api.v2.movement.right()
                time.sleep(1.6)
                robot_api.api.v2.movement.stop()
                robot_api.api.v2.movement.forward()
                time.sleep(2)
                robot_api.api.v2.movement.stop()
                CURRENTLY_RUNNING_PROCEDURE = False
        def turn_left():
            global CURRENTLY_RUNNING_PROCEDURE
            if not CURRENTLY_RUNNING_PROCEDURE:
                CURRENTLY_RUNNING_PROCEDURE = True

                robot_api.api.v2.movement.stop()
                robot_api.api.v2.movement.forward()
                print("Automatically moving forward")
                time.sleep(2.5)
                robot_api.api.v2.movement.stop()
                robot_api.api.v2.movement.left()
                print("Automatically moving left")
                time.sleep(1)
                robot_api.api.v2.movement.stop()
                print("Automatic stop. Procedure finished")

                CURRENTLY_RUNNING_PROCEDURE = False"""

        if not CURRENTLY_RUNNING_PROCEDURE:
            pass
            # threading.Thread(target=turn_right).start()
    
    """
    frame = ORB(frame)
    if not_alone:
       frame = show_not_alone(frame)

    if is_5_percent_blue(frame):
        pass
        # print("Over 5%")
        # robot_api.api.v2.movement.stop()
    else:
        pass
        # print("Less than 5%")
    if lines is not None:
        n_lines = compute_line(lines)
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
    """

    return frame
