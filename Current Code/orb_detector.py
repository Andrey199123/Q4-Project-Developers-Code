# robot_api/camera/orb_detector.py
import cv2
import time
import os.path
import robot_api.api.v2.movement # For robot_api.api.v2.movement.stop()

MARTIAN_PATH = "/home/insomnia/Documents/Andreytesting/robot_api/camera/martian.png"
martian = None
if os.path.exists(MARTIAN_PATH):
    martian = cv2.imread(MARTIAN_PATH, 0)
    if martian is None:
        print(f"Error: Martian image not loaded from {MARTIAN_PATH}. Check path and file integrity.")
else:
    print(f"Error: Martian image path does not exist: {MARTIAN_PATH}")

def show_not_alone_text(frame):
    """
    Puts a "WE ARE NOT ALONE!!" text box on the frame.

    Args:
        frame: The image frame to draw on.

    Returns:
        The frame with the text box.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)  # White color
    thickness = 2
    position = (10, 50)
    text = "WE ARE NOT ALONE!!"
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_width, text_height = text_size
    rectangle_color = (0, 0, 255)  # Red color
    rectangle_position1 = position
    rectangle_position2 = (position[0] + text_width, position[1] - text_height - 10)
    cv2.rectangle(frame, rectangle_position1, rectangle_position2, rectangle_color, -1)
    cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return frame

def detect_martian_orb(frame, not_alone_timer_end):
    """
    Detects and displays the Martian on each frame using ORB.

    Args:
        frame: An image captured from a live video stream.
        not_alone_timer_end (float): Timestamp until which the "not alone" message should persist.

    Returns:
        tuple: (modified_frame, new_not_alone_timer_end, martian_detected_flag)
               modified_frame: The frame, possibly with the "not alone" text.
               new_not_alone_timer_end: Updated timestamp for the message.
               martian_detected_flag: Boolean indicating if the Martian was robustly detected in this frame.
    """
    global martian
    martian_detected_this_frame = False

    if martian is None:
        print("Martian template not loaded, skipping ORB detection.")
        return frame, not_alone_timer_end, martian_detected_this_frame

    frame_blur = cv2.blur(frame, (19, 19))
    frame_gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()

    keypoints1, descriptors1 = orb.detectAndCompute(martian, None)
    keypoints2, descriptors2 = orb.detectAndCompute(frame_gray, None)

    if descriptors1 is None or descriptors2 is None:
        if time.time() < not_alone_timer_end:
            frame = show_not_alone_text(frame)
        return frame, not_alone_timer_end, martian_detected_this_frame

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    try:
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        num_matches = len(matches)
        print(num_matches)
        if num_matches > 10: # Threshold for robust detection
            print("WE ARE NOT ALONE!!!")
            not_alone_timer_end = time.time() + 5  # Show message for 5 seconds
            robot_api.api.v2.movement.stop() # Stop robot on detection
            frame = show_not_alone_text(frame)
            martian_detected_this_frame = True
        elif time.time() < not_alone_timer_end: # Persist message if timer is active
            frame = show_not_alone_text(frame)
        else:
            pass

    except Exception as e:
        print(f"Error in ORB matching: {e}")
        if time.time() < not_alone_timer_end:
            frame = show_not_alone_text(frame)

    return frame, not_alone_timer_end, martian_detected_this_frame
